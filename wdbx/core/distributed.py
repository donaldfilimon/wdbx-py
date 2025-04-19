"""
Distributed architecture support for WDBX.
"""

import os
import logging
import asyncio
import json
import socket
import pickle
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ShardManager:
    """
    Manages distributed shards for WDBX.

    This class handles the distribution of vectors across multiple shards,
    shard replication, and coordination between nodes in a cluster.

    Attributes:
        num_shards (int): Number of shards
        data_dir (Path): Directory for storing shard data
        config (Dict[str, Any]): Configuration dictionary
    """

    def __init__(
        self,
        num_shards: int,
        data_dir: Path,
        config: Any,
        is_coordinator: bool = True,
    ):
        """
        Initialize the shard manager.

        Args:
            num_shards: Number of shards to manage
            data_dir: Directory for storing shard data
            config: Configuration object
            is_coordinator: Whether this node is the coordinator
        """
        self.num_shards = num_shards
        self.data_dir = data_dir
        self.config = config
        self.is_coordinator = is_coordinator

        # Configuration parameters
        self.host = self.config.get("DISTRIBUTED_HOST", "localhost")
        self.port = int(self.config.get("DISTRIBUTED_PORT", 7777))
        self.auth_enabled = self.config.get("DISTRIBUTED_AUTH_ENABLED", False)
        self.auth_key = self.config.get("DISTRIBUTED_AUTH_KEY", "")

        # Shard allocation
        self.shard_allocation = {}  # shard_id -> node_id
        self.shard_replicas = {}  # shard_id -> list of node_ids
        self.replication_factor = int(
            self.config.get("DISTRIBUTED_REPLICATION_FACTOR", 1)
        )

        # Node information
        self.node_id = self._generate_node_id()
        self.nodes = {}  # node_id -> node_info

        # Communication
        self.server = None
        self.client = None

        # Thread pool for background tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"Initialized ShardManager with {num_shards} shards, node_id={self.node_id}"
        )

    def _generate_node_id(self) -> str:
        """Generate a unique node ID based on hostname and timestamp."""
        hostname = socket.gethostname()
        timestamp = int(time.time())
        return f"{hostname}_{timestamp}"

    async def initialize(self):
        """Initialize the distributed architecture."""
        logger.debug("Initializing distributed architecture")

        # Create shard directories
        for shard_id in range(self.num_shards):
            shard_dir = self.data_dir / f"shard_{shard_id}"
            if not shard_dir.exists():
                shard_dir.mkdir(parents=True, exist_ok=True)

        # Load shard allocation if available
        await self._load_shard_allocation()

        # Start server if enabled
        if self.config.get("DISTRIBUTED_SERVER_ENABLED", True):
            await self._start_server()

        # Connect to coordinator if not coordinator
        if not self.is_coordinator:
            await self._connect_to_coordinator()

        # Allocate shards if coordinator
        if self.is_coordinator:
            await self._allocate_shards()

    async def shutdown(self):
        """Clean up resources and shut down distributed components."""
        logger.debug("Shutting down distributed architecture")

        # Stop server
        if self.server:
            await self._stop_server()

        # Disconnect client
        if self.client:
            await self._disconnect_client()

        # Save shard allocation
        await self._save_shard_allocation()

        # Shutdown thread pool
        self.thread_pool.shutdown()

    async def _load_shard_allocation(self):
        """Load shard allocation from disk."""
        allocation_file = self.data_dir / "shard_allocation.json"

        if allocation_file.exists():
            try:
                with open(allocation_file, "r") as f:
                    allocation_data = json.load(f)

                self.shard_allocation = allocation_data.get("allocation", {})
                self.shard_replicas = allocation_data.get("replicas", {})
                self.nodes = allocation_data.get("nodes", {})

                # Convert keys from strings to integers
                self.shard_allocation = {
                    int(k): v for k, v in self.shard_allocation.items()
                }
                self.shard_replicas = {
                    int(k): v for k, v in self.shard_replicas.items()
                }

                logger.debug(
                    f"Loaded shard allocation for {len(self.shard_allocation)} shards"
                )
            except Exception as e:
                logger.error(f"Error loading shard allocation: {e}")
                # Initialize empty allocation
                self.shard_allocation = {}
                self.shard_replicas = {}
                self.nodes = {}

    async def _save_shard_allocation(self):
        """Save shard allocation to disk."""
        allocation_file = self.data_dir / "shard_allocation.json"

        try:
            allocation_data = {
                "allocation": self.shard_allocation,
                "replicas": self.shard_replicas,
                "nodes": self.nodes,
            }

            with open(allocation_file, "w") as f:
                json.dump(allocation_data, f, indent=2)

            logger.debug(
                f"Saved shard allocation for {len(self.shard_allocation)} shards"
            )
        except Exception as e:
            logger.error(f"Error saving shard allocation: {e}")

    async def _start_server(self):
        """Start the server for handling distributed requests."""
        if self.server:
            logger.warning("Server already running")
            return

        try:
            # Create server in a separate process
            ctx = mp.get_context("spawn")
            self.server_queue = ctx.Queue()
            self.server_process = ctx.Process(
                target=self._run_server,
                args=(
                    self.host,
                    self.port,
                    self.auth_enabled,
                    self.auth_key,
                    self.server_queue,
                ),
            )
            self.server_process.start()

            # Wait for server to start
            server_info = self.server_queue.get(timeout=10)
            if server_info.get("status") == "running":
                logger.info(f"Server started on {self.host}:{self.port}")
                self.server = server_info
            else:
                logger.error(f"Failed to start server: {server_info.get('error')}")
                self.server_process.terminate()
                self.server_process = None
        except Exception as e:
            logger.error(f"Error starting server: {e}")

    def _run_server(self, host, port, auth_enabled, auth_key, queue):
        """Run the server in a separate process."""
        import socket
        import selectors
        import struct
        import json

        try:
            # Create server socket
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(128)
            server_socket.setblocking(False)

            # Create selector for non-blocking I/O
            selector = selectors.DefaultSelector()
            selector.register(server_socket, selectors.EVENT_READ, data=None)

            # Inform parent process that server is running
            queue.put({"status": "running", "host": host, "port": port})

            # Main server loop
            running = True
            while running:
                events = selector.select(timeout=1)
                for key, mask in events:
                    if key.data is None:
                        # New connection
                        client_socket, addr = server_socket.accept()
                        client_socket.setblocking(False)
                        selector.register(
                            client_socket,
                            selectors.EVENT_READ,
                            data={"addr": addr, "buffer": b""},
                        )
                    else:
                        # Existing connection
                        client_socket = key.fileobj
                        data = key.data

                        try:
                            # Read data
                            chunk = client_socket.recv(4096)
                            if chunk:
                                data["buffer"] += chunk

                                # Process complete messages
                                while len(data["buffer"]) >= 4:
                                    # Get message length
                                    msg_len = struct.unpack("!I", data["buffer"][:4])[0]

                                    # Check if we have a complete message
                                    if len(data["buffer"]) >= msg_len + 4:
                                        # Extract message
                                        message = data["buffer"][4 : msg_len + 4]
                                        data["buffer"] = data["buffer"][msg_len + 4 :]

                                        # Process message
                                        try:
                                            decoded_message = pickle.loads(message)

                                            # Handle authentication
                                            if auth_enabled and not data.get(
                                                "authenticated", False
                                            ):
                                                if (
                                                    decoded_message.get("type")
                                                    == "auth"
                                                    and decoded_message.get("key")
                                                    == auth_key
                                                ):
                                                    data["authenticated"] = True
                                                    response = {
                                                        "type": "auth_response",
                                                        "status": "success",
                                                    }
                                                else:
                                                    response = {
                                                        "type": "auth_response",
                                                        "status": "failure",
                                                        "error": "Authentication failed",
                                                    }

                                                # Send response
                                                response_bytes = pickle.dumps(response)
                                                header = struct.pack(
                                                    "!I", len(response_bytes)
                                                )
                                                client_socket.sendall(
                                                    header + response_bytes
                                                )

                                                # If authentication failed, close connection
                                                if response["status"] == "failure":
                                                    selector.unregister(client_socket)
                                                    client_socket.close()
                                                    continue

                                            # If authentication is required but client is not authenticated
                                            elif auth_enabled and not data.get(
                                                "authenticated", False
                                            ):
                                                # Send error response
                                                response = {
                                                    "type": "error",
                                                    "error": "Authentication required",
                                                }
                                                response_bytes = pickle.dumps(response)
                                                header = struct.pack(
                                                    "!I", len(response_bytes)
                                                )
                                                client_socket.sendall(
                                                    header + response_bytes
                                                )

                                                # Close connection
                                                selector.unregister(client_socket)
                                                client_socket.close()
                                                continue

                                            # Handle message
                                            if decoded_message.get("type") == "ping":
                                                response = {"type": "pong"}
                                            elif (
                                                decoded_message.get("type")
                                                == "shutdown"
                                            ):
                                                response = {
                                                    "type": "shutdown_response",
                                                    "status": "success",
                                                }
                                                running = False
                                            else:
                                                # Unknown message type
                                                response = {
                                                    "type": "error",
                                                    "error": f"Unknown message type: {decoded_message.get('type')}",
                                                }

                                            # Send response
                                            response_bytes = pickle.dumps(response)
                                            header = struct.pack(
                                                "!I", len(response_bytes)
                                            )
                                            client_socket.sendall(
                                                header + response_bytes
                                            )
                                        except Exception as e:
                                            # Send error response
                                            response = {
                                                "type": "error",
                                                "error": str(e),
                                            }
                                            response_bytes = pickle.dumps(response)
                                            header = struct.pack(
                                                "!I", len(response_bytes)
                                            )
                                            client_socket.sendall(
                                                header + response_bytes
                                            )
                            else:
                                # Connection closed by client
                                selector.unregister(client_socket)
                                client_socket.close()
                        except Exception as e:
                            # Error handling client
                            selector.unregister(client_socket)
                            client_socket.close()
                            print(f"Error handling client: {e}")

            # Clean up
            selector.close()
            server_socket.close()
        except Exception as e:
            # Inform parent process of error
            queue.put({"status": "error", "error": str(e)})

    async def _stop_server(self):
        """Stop the server."""
        if not self.server:
            return

        try:
            # Connect to server and send shutdown command
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))

            # Send shutdown message
            message = {"type": "shutdown"}
            message_bytes = pickle.dumps(message)
            header = struct.pack("!I", len(message_bytes))
            client_socket.sendall(header + message_bytes)

            # Wait for response
            header = client_socket.recv(4)
            msg_len = struct.unpack("!I", header)[0]
            message = client_socket.recv(msg_len)
            response = pickle.loads(message)

            # Close connection
            client_socket.close()

            # Wait for server process to terminate
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                self.server_process.terminate()

            self.server = None
            self.server_process = None
            logger.info("Server stopped")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            # Force terminate server process
            if (
                hasattr(self, "server_process")
                and self.server_process
                and self.server_process.is_alive()
            ):
                self.server_process.terminate()
            self.server = None
            self.server_process = None

    async def _connect_to_coordinator(self):
        """Connect to the coordinator node."""
        if self.client:
            logger.warning("Already connected to coordinator")
            return

        try:
            # Get coordinator address
            coordinator_host = self.config.get(
                "DISTRIBUTED_COORDINATOR_HOST", "localhost"
            )
            coordinator_port = int(
                self.config.get("DISTRIBUTED_COORDINATOR_PORT", 7777)
            )

            # Create client socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((coordinator_host, coordinator_port))

            # Authenticate if required
            if self.auth_enabled:
                auth_message = {"type": "auth", "key": self.auth_key}
                auth_bytes = pickle.dumps(auth_message)
                header = struct.pack("!I", len(auth_bytes))
                client_socket.sendall(header + auth_bytes)

                # Wait for response
                header = client_socket.recv(4)
                msg_len = struct.unpack("!I", header)[0]
                message = client_socket.recv(msg_len)
                response = pickle.loads(message)

                if response.get("status") != "success":
                    raise ValueError(f"Authentication failed: {response.get('error')}")

            # Register with coordinator
            register_message = {
                "type": "register",
                "node_id": self.node_id,
                "host": self.host,
                "port": self.port,
                "capabilities": {
                    "storage": True,
                    "compute": True,
                },
            }
            register_bytes = pickle.dumps(register_message)
            header = struct.pack("!I", len(register_bytes))
            client_socket.sendall(header + register_bytes)

            # Wait for response
            header = client_socket.recv(4)
            msg_len = struct.unpack("!I", header)[0]
            message = client_socket.recv(msg_len)
            response = pickle.loads(message)

            if response.get("status") != "success":
                raise ValueError(f"Registration failed: {response.get('error')}")

            # Store client socket
            self.client = {
                "socket": client_socket,
                "host": coordinator_host,
                "port": coordinator_port,
            }

            logger.info(
                f"Connected to coordinator at {coordinator_host}:{coordinator_port}"
            )
        except Exception as e:
            logger.error(f"Error connecting to coordinator: {e}")
            if hasattr(self, "client") and self.client and self.client.get("socket"):
                self.client["socket"].close()
            self.client = None

    async def _disconnect_client(self):
        """Disconnect from the coordinator."""
        if not self.client:
            return

        try:
            # Send unregister message
            unregister_message = {"type": "unregister", "node_id": self.node_id}
            unregister_bytes = pickle.dumps(unregister_message)
            header = struct.pack("!I", len(unregister_bytes))
            self.client["socket"].sendall(header + unregister_bytes)

            # Wait for response
            header = self.client["socket"].recv(4)
            msg_len = struct.unpack("!I", header)[0]
            message = self.client["socket"].recv(msg_len)
            response = pickle.loads(message)

            # Close connection
            self.client["socket"].close()
            self.client = None
            logger.info("Disconnected from coordinator")
        except Exception as e:
            logger.error(f"Error disconnecting from coordinator: {e}")
            # Force close socket
            if hasattr(self, "client") and self.client and self.client.get("socket"):
                try:
                    self.client["socket"].close()
                except:
                    pass
            self.client = None

    async def _allocate_shards(self):
        """Allocate shards to available nodes."""
        if not self.is_coordinator:
            logger.warning("Not a coordinator, cannot allocate shards")
            return

        # Get active nodes
        active_nodes = [
            node_id
            for node_id, node in self.nodes.items()
            if node.get("status") == "active"
        ]

        # If no active nodes, allocate to self
        if not active_nodes:
            active_nodes = [self.node_id]
            self.nodes[self.node_id] = {
                "status": "active",
                "host": self.host,
                "port": self.port,
                "capabilities": {
                    "storage": True,
                    "compute": True,
                },
            }

        # Allocate primary shards
        for shard_id in range(self.num_shards):
            # If shard is already allocated, skip
            if shard_id in self.shard_allocation:
                continue

            # Allocate to node with fewest shards
            node_shard_counts = {}
            for node_id in active_nodes:
                node_shard_counts[node_id] = 0

            for allocated_shard, node_id in self.shard_allocation.items():
                if node_id in node_shard_counts:
                    node_shard_counts[node_id] += 1

            # Find node with fewest shards
            min_shards = float("inf")
            min_node = None
            for node_id, count in node_shard_counts.items():
                if count < min_shards:
                    min_shards = count
                    min_node = node_id

            # Allocate shard to node
            if min_node:
                self.shard_allocation[shard_id] = min_node
                logger.debug(f"Allocated shard {shard_id} to node {min_node}")

        # Allocate replica shards
        for shard_id in range(self.num_shards):
            # Initialize replica list if not exists
            if shard_id not in self.shard_replicas:
                self.shard_replicas[shard_id] = []

            # Primary node
            primary_node = self.shard_allocation.get(shard_id)
            if not primary_node:
                continue

            # Allocate replicas
            for replica_idx in range(self.replication_factor):
                # If replica already allocated, skip
                if len(self.shard_replicas[shard_id]) > replica_idx:
                    continue

                # Find eligible nodes (not primary and not already a replica)
                eligible_nodes = [
                    node_id
                    for node_id in active_nodes
                    if node_id != primary_node
                    and node_id not in self.shard_replicas[shard_id]
                ]

                if not eligible_nodes:
                    break

                # Allocate to node with fewest replicas
                node_replica_counts = {}
                for node_id in eligible_nodes:
                    node_replica_counts[node_id] = 0

                for shard_replicas in self.shard_replicas.values():
                    for node_id in shard_replicas:
                        if node_id in node_replica_counts:
                            node_replica_counts[node_id] += 1

                # Find node with fewest replicas
                min_replicas = float("inf")
                min_node = None
                for node_id, count in node_replica_counts.items():
                    if count < min_replicas:
                        min_replicas = count
                        min_node = node_id

                # Allocate replica to node
                if min_node:
                    self.shard_replicas[shard_id].append(min_node)
                    logger.debug(
                        f"Allocated replica for shard {shard_id} to node {min_node}"
                    )

        # Save allocation
        await self._save_shard_allocation()

    def get_shard_info(self, shard_id: int) -> Dict[str, Any]:
        """
        Get information about a shard.

        Args:
            shard_id: ID of the shard

        Returns:
            Dictionary with shard information
        """
        if shard_id < 0 or shard_id >= self.num_shards:
            raise ValueError(f"Invalid shard ID: {shard_id}")

        primary_node = self.shard_allocation.get(shard_id)
        replica_nodes = self.shard_replicas.get(shard_id, [])

        return {
            "shard_id": shard_id,
            "primary_node": primary_node,
            "replica_nodes": replica_nodes,
            "primary_node_info": self.nodes.get(primary_node, {}),
            "replica_node_info": [self.nodes.get(node, {}) for node in replica_nodes],
        }

    def is_local_shard(self, shard_id: int) -> bool:
        """
        Check if a shard is local to this node.

        Args:
            shard_id: ID of the shard

        Returns:
            True if the shard is local, False otherwise
        """
        if shard_id < 0 or shard_id >= self.num_shards:
            raise ValueError(f"Invalid shard ID: {shard_id}")

        primary_node = self.shard_allocation.get(shard_id)
        replica_nodes = self.shard_replicas.get(shard_id, [])

        return primary_node == self.node_id or self.node_id in replica_nodes

    async def forward_request(
        self, shard_id: int, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward a request to the node responsible for a shard.

        Args:
            shard_id: ID of the shard
            request: Request to forward

        Returns:
            Response from the node
        """
        # Check if shard is local
        if self.is_local_shard(shard_id):
            logger.debug(f"Request for shard {shard_id} is local")
            # Handle locally
            return {"status": "success", "local": True}

        # Get shard info
        shard_info = self.get_shard_info(shard_id)
        primary_node = shard_info["primary_node"]

        if not primary_node:
            raise ValueError(f"No primary node for shard {shard_id}")

        primary_node_info = shard_info["primary_node_info"]
        if not primary_node_info:
            raise ValueError(f"No information for node {primary_node}")

        # Forward request to primary node
        try:
            logger.debug(
                f"Forwarding request for shard {shard_id} to node {primary_node}"
            )
            # Create client socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(
                (primary_node_info["host"], primary_node_info["port"])
            )

            # Authenticate if required
            if self.auth_enabled:
                auth_message = {"type": "auth", "key": self.auth_key}
                auth_bytes = pickle.dumps(auth_message)
                header = struct.pack("!I", len(auth_bytes))
                client_socket.sendall(header + auth_bytes)

                # Wait for response
                header = client_socket.recv(4)
                msg_len = struct.unpack("!I", header)[0]
                message = client_socket.recv(msg_len)
                response = pickle.loads(message)

                if response.get("status") != "success":
                    raise ValueError(f"Authentication failed: {response.get('error')}")

            # Send request
            request_message = {
                "type": "shard_request",
                "shard_id": shard_id,
                "request": request,
            }
            request_bytes = pickle.dumps(request_message)
            header = struct.pack("!I", len(request_bytes))
            client_socket.sendall(header + request_bytes)

            # Wait for response
            header = client_socket.recv(4)
            msg_len = struct.unpack("!I", header)[0]
            message = client_socket.recv(msg_len)
            response = pickle.loads(message)

            # Close connection
            client_socket.close()

            return response
        except Exception as e:
            logger.error(f"Error forwarding request to node {primary_node}: {e}")

            # Try replicas
            for replica_node, replica_info in zip(
                shard_info["replica_nodes"], shard_info["replica_node_info"]
            ):
                try:
                    logger.debug(
                        f"Trying replica node {replica_node} for shard {shard_id}"
                    )
                    # Create client socket
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket.connect((replica_info["host"], replica_info["port"]))

                    # Authenticate if required
                    if self.auth_enabled:
                        auth_message = {"type": "auth", "key": self.auth_key}
                        auth_bytes = pickle.dumps(auth_message)
                        header = struct.pack("!I", len(auth_bytes))
                        client_socket.sendall(header + auth_bytes)

                        # Wait for response
                        header = client_socket.recv(4)
                        msg_len = struct.unpack("!I", header)[0]
                        message = client_socket.recv(msg_len)
                        response = pickle.loads(message)

                        if response.get("status") != "success":
                            raise ValueError(
                                f"Authentication failed: {response.get('error')}"
                            )

                    # Send request
                    request_message = {
                        "type": "shard_request",
                        "shard_id": shard_id,
                        "request": request,
                    }
                    request_bytes = pickle.dumps(request_message)
                    header = struct.pack("!I", len(request_bytes))
                    client_socket.sendall(header + request_bytes)

                    # Wait for response
                    header = client_socket.recv(4)
                    msg_len = struct.unpack("!I", header)[0]
                    message = client_socket.recv(msg_len)
                    response = pickle.loads(message)

                    # Close connection
                    client_socket.close()

                    return response
                except Exception as replica_error:
                    logger.error(
                        f"Error forwarding request to replica node {replica_node}: {replica_error}"
                    )

            # All nodes failed
            raise ValueError(f"Failed to forward request for shard {shard_id}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the distributed architecture.

        Returns:
            Dictionary with statistics
        """
        return {
            "node_id": self.node_id,
            "is_coordinator": self.is_coordinator,
            "num_shards": self.num_shards,
            "replication_factor": self.replication_factor,
            "num_nodes": len(self.nodes),
            "active_nodes": len(
                [
                    node
                    for node, info in self.nodes.items()
                    if info.get("status") == "active"
                ]
            ),
            "shards_per_node": self._get_shards_per_node(),
        }

    def _get_shards_per_node(self) -> Dict[str, Dict[str, int]]:
        """Get the number of primary and replica shards per node."""
        result = {}

        for node_id in self.nodes:
            result[node_id] = {"primary": 0, "replica": 0}

        for shard_id, node_id in self.shard_allocation.items():
            if node_id in result:
                result[node_id]["primary"] += 1

        for shard_id, replicas in self.shard_replicas.items():
            for node_id in replicas:
                if node_id in result:
                    result[node_id]["replica"] += 1

        return result
