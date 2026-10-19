"""WebSocket control server (optional, off by default).

Lets a phone / web dashboard drive GestureOS and receive live gesture events:

* **Inbound**  ``{"command": "click", "params": {...}}`` -> dispatched to the
  callback registered by the engine (``main.py::_on_network_command``).
* **Outbound** ``broadcast({...})`` pushes gesture telemetry to every client.

The asyncio loop lives on its own thread; :meth:`broadcast` and :meth:`stop` are
safe to call from the gesture thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Set

from utils import module_available, try_import

logger = logging.getLogger(__name__)

DEFAULT_NETWORK_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "host": "0.0.0.0",
    "port": 8765,
    "heartbeat_interval": 5.0,
    "max_clients": 8,
    "token": "",                    # when set, clients must send {"token": "..."}
    "ping_interval": 20.0,
    "queue_size": 256,
}


class GestureNetworkServer:
    """WebSocket server exposing gesture telemetry and remote commands."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(config or {})
        self.config: Dict[str, Any] = {**DEFAULT_NETWORK_CONFIG, **config}
        # main.py assigns these directly before starting the loop.
        self._host: str = str(self.config.get("host", "0.0.0.0"))
        self._port: int = int(self.config.get("port", 8765))
        self._callback: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._clients: Set[Any] = set()
        self._clients_lock = threading.Lock()
        self._running = False
        self._broadcast_queue: "asyncio.Queue | None" = None
        self.stats: Dict[str, Any] = {
            "started_at": None,
            "commands_received": 0,
            "messages_sent": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------ #
    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def is_available(self) -> bool:
        return module_available("websockets")

    def set_callback(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        self._callback = callback

    # ------------------------------------------------------------------ #
    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        """Run the server (blocking - ``main.py`` calls this on a worker thread)."""
        if host:
            self._host = str(host)
        if port:
            self._port = int(port)
        if not self.is_available():
            logger.error("The 'websockets' package is required for the network server")
            return False

        asyncio.run(self._serve())
        return True

    def start_async(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        """Start the server on its own thread and return immediately."""
        if self._running:
            return True
        self._thread = threading.Thread(
            target=self.start, args=(host, port), name="GestureOS-Network", daemon=True
        )
        self._thread.start()
        deadline = time.time() + 5.0
        while not self._running and time.time() < deadline:
            time.sleep(0.05)
        return self._running

    def stop(self) -> None:
        self._running = False
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        with self._clients_lock:
            self._clients.clear()
        logger.info("Network server stopped")

    # ------------------------------------------------------------------ #
    async def _serve(self) -> None:
        websockets = try_import("websockets")
        if websockets is None:
            return
        serve = getattr(websockets, "serve", None) or getattr(websockets, "Server", None)
        if serve is None:  # pragma: no cover
            logger.error("Unsupported websockets version")
            return

        self._loop = asyncio.get_running_loop()
        self._broadcast_queue = asyncio.Queue(maxsize=int(self.config.get("queue_size", 256)))
        self._running = True
        self.stats["started_at"] = time.time()

        kwargs: Dict[str, Any] = {}
        if "ping_interval" in self.config:
            kwargs["ping_interval"] = float(self.config.get("ping_interval", 20.0))

        try:
            async with serve(self._handler, self._host, self._port, **kwargs) as server:
                self._server = server
                logger.info("GestureOS network server listening on ws://%s:%s", self._host, self._port)
                broadcaster = asyncio.create_task(self._broadcaster())
                heartbeats = asyncio.create_task(self._heartbeat())
                try:
                    await asyncio.Future()  # run until stopped
                finally:
                    broadcaster.cancel()
                    heartbeats.cancel()
        except OSError as exc:
            logger.error("Network server could not bind %s:%s - %s", self._host, self._port, exc)
            self.stats["errors"] += 1
        except asyncio.CancelledError:  # pragma: no cover
            pass
        finally:
            self._running = False
            self._server = None

    async def _handler(self, websocket, path: Optional[str] = None) -> None:
        with self._clients_lock:
            if len(self._clients) >= int(self.config.get("max_clients", 8)):
                logger.warning("Rejecting client: max_clients reached")
                await websocket.close(code=1013, reason="too many clients")
                return
            self._clients.add(websocket)
        logger.info("Network client connected (%d total)", self.client_count)

        token = str(self.config.get("token", "") or "")
        try:
            await websocket.send(json.dumps({
                "type": "hello",
                "service": "GestureOS",
                "version": 1,
                "auth_required": bool(token),
                "t": time.time(),
            }))
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    await websocket.send(json.dumps({"type": "error", "error": "invalid json"}))
                    continue
                if not isinstance(message, dict):
                    continue
                if token and message.get("token") != token:
                    await websocket.send(json.dumps({"type": "error", "error": "bad token"}))
                    continue
                await self._dispatch(message, websocket)
        except Exception as exc:
            logger.debug("Network client error: %s", exc)
            self.stats["errors"] += 1
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)
            logger.info("Network client disconnected (%d total)", self.client_count)

    async def _dispatch(self, message: Dict[str, Any], websocket) -> None:
        kind = str(message.get("type", "command"))
        if kind == "ping":
            await self._send(websocket, {"type": "pong", "t": time.time()})
            return
        if kind == "status":
            await self._send(websocket, {"type": "status", **self.status()})
            return

        self.stats["commands_received"] += 1
        callback = self._callback
        if callback is None:
            await self._send(websocket, {"type": "error", "error": "no command handler registered"})
            return
        try:
            result = callback(message)
            if asyncio.iscoroutine(result):
                result = await result
            await self._send(websocket, {"type": "ack", "command": message.get("command"), "result": bool(result)})
        except Exception as exc:
            logger.error("Command dispatch failed: %s", exc)
            self.stats["errors"] += 1
            await self._send(websocket, {"type": "error", "error": str(exc)})

    async def _send(self, websocket, payload: Dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(payload, default=str))
            self.stats["messages_sent"] += 1
        except Exception as exc:
            logger.debug("Send failed: %s", exc)
            self.stats["errors"] += 1

    async def _broadcaster(self) -> None:
        while self._running:
            payload = await self._broadcast_queue.get()
            with self._clients_lock:
                clients = list(self._clients)
            for client in clients:
                await self._send(client, payload)

    async def _heartbeat(self) -> None:
        interval = float(self.config.get("heartbeat_interval", 5.0))
        if interval <= 0:
            return
        while self._running:
            await asyncio.sleep(interval)
            if self.client_count:
                await self._enqueue({"type": "heartbeat", **self.status()})

    async def _enqueue(self, payload: Dict[str, Any]) -> None:
        if self._broadcast_queue is None:
            return
        try:
            self._broadcast_queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.debug("Broadcast queue full - dropping message")

    # ------------------------------------------------------------------ #
    def broadcast(self, payload: Dict[str, Any]) -> bool:
        """Queue ``payload`` for every connected client (thread safe)."""
        loop = self._loop
        if loop is None or not self._running:
            return False
        data = {"type": "gesture", "t": time.time(), **(payload or {})}
        try:
            asyncio.run_coroutine_threadsafe(self._enqueue(data), loop)
            return True
        except Exception as exc:
            logger.debug("Broadcast failed: %s", exc)
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "host": self._host,
            "port": self._port,
            "clients": self.client_count,
            **{k: v for k, v in self.stats.items()},
        }


__all__ = ["GestureNetworkServer", "DEFAULT_NETWORK_CONFIG"]
