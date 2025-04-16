"""
Command-line interface for WDBX.

This module provides a CLI for interacting with WDBX.
"""

import os
import sys
import logging
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from pathlib import Path

from .core.wdbx import WDBX
from .core.config import WDBXConfig

logger = logging.getLogger(__name__)


class WDBXCLI:
    """
    Command-line interface for WDBX.

    This class provides a CLI for interacting with WDBX and its plugins.

    Attributes:
        wdbx: Reference to the WDBX instance
        commands: Dictionary of registered commands
    """

    def __init__(self, wdbx):
        """
        Initialize the CLI.

        Args:
            wdbx: Reference to the WDBX instance
        """
        self.wdbx = wdbx
        self.commands = {}

        # Register built-in commands
        self._register_builtin_commands()

        # Register plugin commands
        self._register_plugin_commands()

        logger.debug(f"Initialized WDBXCLI with {len(self.commands)} commands")

    def _register_builtin_commands(self):
        """Register built-in commands."""
        # Help command
        self.register_command(
            "help",
            self._cmd_help,
            "Display help information",
            {
                "command": "Command to get help for",
            }
        )

        # Version command
        self.register_command(
            "version",
            self._cmd_version,
            "Display version information",
            {}
        )

        # Store command
        self.register_command(
            "store", self._cmd_store, "Store a vector",
            {"--id": "Vector ID", "--vector": "Vector data (JSON array)",
             "--metadata": "Metadata (JSON object)",
             "--from-file": "Load vector from file",
             "--from-text": "Create vector from text using an embedding plugin", })

        # Search command
        self.register_command(
            "search", self._cmd_search, "Search for similar vectors",
            {"--vector": "Query vector (JSON array)",
             "--from-file": "Load query vector from file",
             "--from-text":
             "Create query vector from text using an embedding plugin",
             "--limit": "Maximum number of results",
             "--threshold": "Minimum similarity threshold",
             "--filter": "Metadata filter (JSON object)", })

        # Get command
        self.register_command(
            "get",
            self._cmd_get,
            "Get a vector by ID",
            {
                "id": "Vector ID",
            }
        )

        # Delete command
        self.register_command(
            "delete",
            self._cmd_delete,
            "Delete a vector",
            {
                "id": "Vector ID",
            }
        )

        # Update metadata command
        self.register_command(
            "update-metadata",
            self._cmd_update_metadata,
            "Update vector metadata",
            {
                "id": "Vector ID",
                "--metadata": "New metadata (JSON object)",
                "--from-file": "Load metadata from file",
            }
        )

        # Stats command
        self.register_command(
            "stats",
            self._cmd_stats,
            "Display database statistics",
            {}
        )

        # Clear command
        self.register_command(
            "clear",
            self._cmd_clear,
            "Clear the database",
            {
                "--force": "Skip confirmation prompt",
            }
        )

        # List plugins command
        self.register_command(
            "plugins",
            self._cmd_plugins,
            "List available plugins",
            {}
        )

        # Plugin info command
        self.register_command(
            "plugin-info",
            self._cmd_plugin_info,
            "Display plugin information",
            {
                "name": "Plugin name",
            }
        )

        # Start API server command
        self.register_command(
            "serve",
            self._cmd_serve,
            "Start the API server",
            {
                "--host": "Host to bind to",
                "--port": "Port to listen on",
                "--auth": "Enable authentication",
                "--auth-key": "Authentication key",
                "--cors": "Enable CORS",
                "--cors-origins": "Allowed CORS origins (comma-separated)",
            }
        )

    def _register_plugin_commands(self):
        """Register commands from plugins."""
        for plugin_name, plugin in self.wdbx.plugins.items():
            if hasattr(plugin, "register_commands"):
                try:
                    plugin.register_commands()
                except Exception as e:
                    logger.error(
                        f"Error registering commands for plugin {plugin_name}: {e}")

    def register_command(
            self, name: str, handler, description: str,
            options: Dict[str, str] = None):
        """
        Register a command.

        Args:
            name: Command name
            handler: Command handler function
            description: Command description
            options: Dictionary of option descriptions
        """
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "options": options or {},
        }
        logger.debug(f"Registered command: {name}")

    async def run_command(self, command: str, args: str = ""):
        """
        Run a command.

        Args:
            command: Command name
            args: Command arguments

        Returns:
            Command result

        Raises:
            ValueError: If the command is not found
        """
        if command not in self.commands:
            raise ValueError(f"Command not found: {command}")

        handler = self.commands[command]["handler"]
        try:
            return await handler(args)
        except Exception as e:
            logger.error(f"Error running command '{command}': {e}")
            raise

    async def run_interactive(self):
        """Run in interactive mode."""
        print(f"WDBX CLI v{self.wdbx.version}")
        print("Type 'help' for a list of commands, or 'exit' to quit.")

        while True:
            try:
                # Get command
                cmd_line = input("> ").strip()

                # Exit
                if cmd_line.lower() in ["exit", "quit", "q"]:
                    break

                # Skip empty lines
                if not cmd_line:
                    continue

                # Parse command and arguments
                parts = cmd_line.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                # Run command
                await self.run_command(command, args)

            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                print(f"Error: {e}")

    async def run(self, args: List[str]):
        """
        Run the CLI with command-line arguments.

        Args:
            args: Command-line arguments
        """
        parser = argparse.ArgumentParser(
            description="WDBX Command-line Interface")
        parser.add_argument(
            "--version", action="store_true",
            help="Display version information")
        parser.add_argument("--debug", action="store_true",
                            help="Enable debug logging")

        # Add subparsers for commands
        subparsers = parser.add_subparsers(dest="command", help="Command")

        # Add each command as a subparser
        for cmd_name, cmd_info in self.commands.items():
            cmd_parser = subparsers.add_parser(
                cmd_name, help=cmd_info["description"])

            # Add command options
            for opt_name, opt_desc in cmd_info["options"].items():
                if opt_name.startswith("--"):
                    # Flag or optional argument
                    cmd_parser.add_argument(opt_name, help=opt_desc)
                else:
                    # Positional argument
                    cmd_parser.add_argument(opt_name, help=opt_desc)

        # Parse arguments
        try:
            parsed_args = parser.parse_args(args)
        except argparse.ArgumentError as e:
            logger.error(f"Argument parsing error: {e}")
            print(f"Error: {e}")
            return

        # Set up logging
        log_level = logging.DEBUG if parsed_args.debug else logging.INFO
        logging.basicConfig(
            level=log_level, format="%(levelname)s: %(message)s")

        # Handle version flag
        if parsed_args.version:
            print(f"WDBX v{self.wdbx.version}")
            return

        # Run command
        if parsed_args.command:
            # Convert namespace to string arguments
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if key not in [
                        "command", "version", "debug"] and value is not None:
                    if isinstance(value, bool) and value:
                        cmd_args.append(f"--{key}")
                    else:
                        cmd_args.append(f"--{key} {value}")

            try:
                await self.run_command(parsed_args.command, " ".join(cmd_args))
            except ValueError as e:
                print(f"Error: {e}")
        else:
            # Interactive mode
            await self.run_interactive()

    # Command handlers

    async def _cmd_help(self, args: str):
        """Command handler for the help command."""
        parts = args.split()
        if parts and parts[0] in self.commands:
            # Help for specific command
            cmd_name = parts[0]
            cmd_info = self.commands[cmd_name]
            print(f"{cmd_name}: {cmd_info['description']}")

            if cmd_info["options"]:
                print("\nOptions:")
                for opt_name, opt_desc in cmd_info["options"].items():
                    print(f"  {opt_name}: {opt_desc}")
        else:
            # General help
            print("Available commands:")
            for cmd_name, cmd_info in self.commands.items():
                print(f"  {cmd_name}: {cmd_info['description']}")
            print(
                "\nType 'help <command>' for more information about a specific command.")

    async def _cmd_version(self, args: str):
        """Command handler for the version command."""
        print(f"WDBX v{self.wdbx.version}")

        # Print plugin versions
        if self.wdbx.plugins:
            print("\nPlugins:")
            for plugin_name, plugin in self.wdbx.plugins.items():
                print(f"  {plugin_name}: v{plugin.version}")

    async def _cmd_store(self, args: str):
        """Command handler for the store command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(description="Store a vector")
        parser.add_argument("--id", help="Vector ID")
        parser.add_argument("--vector", help="Vector data (JSON array)")
        parser.add_argument("--metadata", help="Metadata (JSON object)")
        parser.add_argument("--from-file", help="Load vector from file")
        parser.add_argument(
            "--from-text",
            help="Create vector from text using an embedding plugin")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return
        except argparse.ArgumentError as e:
            logger.error(f"Argument parsing error: {e}")
            print(f"Error: {e}")
            return

        # Get vector data
        vector = None

        if parsed_args.vector:
            # Parse JSON vector
            try:
                vector = json.loads(parsed_args.vector)
            except json.JSONDecodeError as e:
                print(f"Error parsing vector: {e}")
                return

        elif parsed_args.from_file:
            # Load vector from file
            try:
                with open(parsed_args.from_file, 'r') as f:
                    vector = json.load(f)
            except Exception as e:
                print(f"Error loading vector from file: {e}")
                return

        elif parsed_args.from_text:
            # Create vector from text using an embedding plugin
            for plugin_name in [
                    "openai", "ollama", "huggingface", "sentencetransformers"]:
                if plugin_name in self.wdbx.plugins:
                    plugin = self.wdbx.plugins[plugin_name]
                    try:
                        vector = await plugin.create_embedding(parsed_args.from_text)
                        break
                    except Exception as e:
                        print(
                            f"Error creating embedding with {plugin_name}: {e}")
                        continue

            if vector is None:
                print("No embedding plugin available")
                return

        else:
            print("No vector data provided")
            return

        # Get metadata
        metadata = None

        if parsed_args.metadata:
            # Parse JSON metadata
            try:
                metadata = json.loads(parsed_args.metadata)
            except json.JSONDecodeError as e:
                print(f"Error parsing metadata: {e}")
                return

        # Store vector
        try:
            vector_id = await self.wdbx.vector_store_async(vector, metadata, parsed_args.id)
            print(f"Vector stored with ID: {vector_id}")
        except Exception as e:
            print(f"Error storing vector: {e}")

    async def _cmd_search(self, args: str):
        """Command handler for the search command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Search for similar vectors")
        parser.add_argument("--vector", help="Query vector (JSON array)")
        parser.add_argument("--from-file", help="Load query vector from file")
        parser.add_argument(
            "--from-text",
            help="Create query vector from text using an embedding plugin")
        parser.add_argument("--limit", type=int, default=10,
                            help="Maximum number of results")
        parser.add_argument(
            "--threshold", type=float, default=0.0,
            help="Minimum similarity threshold")
        parser.add_argument("--filter", help="Metadata filter (JSON object)")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return
        except argparse.ArgumentError as e:
            logger.error(f"Argument parsing error: {e}")
            print(f"Error: {e}")
            return

        # Get query vector
        query_vector = None

        if parsed_args.vector:
            # Parse JSON vector
            try:
                query_vector = json.loads(parsed_args.vector)
            except json.JSONDecodeError as e:
                print(f"Error parsing query vector: {e}")
                return

        elif parsed_args.from_file:
            # Load vector from file
            try:
                with open(parsed_args.from_file, 'r') as f:
                    query_vector = json.load(f)
            except Exception as e:
                print(f"Error loading query vector from file: {e}")
                return

        elif parsed_args.from_text:
            # Create vector from text using an embedding plugin
            for plugin_name in [
                    "openai", "ollama", "huggingface", "sentencetransformers"]:
                if plugin_name in self.wdbx.plugins:
                    plugin = self.wdbx.plugins[plugin_name]
                    try:
                        query_vector = await plugin.create_embedding(parsed_args.from_text)
                        break
                    except Exception as e:
                        print(
                            f"Error creating embedding with {plugin_name}: {e}")

            if query_vector is None:
                print("No embedding plugin available")
                return

        else:
            print("No query vector provided")
            return

        # Get filter
        filter_metadata = None

        if parsed_args.filter:
            # Parse JSON filter
            try:
                filter_metadata = json.loads(parsed_args.filter)
            except json.JSONDecodeError as e:
                print(f"Error parsing filter: {e}")
                return

        # Search vectors
        try:
            results = await self.wdbx.vector_search_async(
                query_vector,
                parsed_args.limit,
                parsed_args.threshold,
                filter_metadata
            )

            print(f"Found {len(results)} results:")
            for vector_id, similarity, metadata in results:
                print(f"  {vector_id} (similarity: {similarity:.4f})")
                if metadata:
                    # Print selected metadata fields for brevity
                    brief_metadata = {}
                    for key, value in metadata.items():
                        if key in ["source", "content", "url", "title",
                                   "description"]:
                            # Truncate long values
                            if isinstance(value, str) and len(value) > 50:
                                brief_metadata[key] = value[:50] + "..."
                            else:
                                brief_metadata[key] = value

                    print(f"    Metadata: {json.dumps(brief_metadata)}")
        except Exception as e:
            print(f"Error searching vectors: {e}")

    async def _cmd_get(self, args: str):
        """Command handler for the get command."""
        if not args:
            print("Vector ID required")
            return

        vector_id = args.strip()

        try:
            result = await self.wdbx.get_vector_async(vector_id)
            if result is None:
                print(f"Vector not found: {vector_id}")
                return

            vector, metadata = result
            print(f"Vector ID: {vector_id}")
            print(f"Vector dimension: {len(vector)}")
            print(f"Vector (truncated): {vector[:5]}...")

            if metadata:
                print("Metadata:")
                for key, value in metadata.items():
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error getting vector: {e}")

    async def _cmd_delete(self, args: str):
        """Command handler for the delete command."""
        if not args:
            print("Vector ID required")
            return

        vector_id = args.strip()

        try:
            success = await self.wdbx.delete_vector_async(vector_id)
            if success:
                print(f"Vector deleted: {vector_id}")
            else:
                print(f"Vector not found: {vector_id}")
        except Exception as e:
            print(f"Error deleting vector: {e}")

    async def _cmd_update_metadata(self, args: str):
        """Command handler for the update-metadata command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(description="Update vector metadata")
        parser.add_argument("id", help="Vector ID")
        parser.add_argument("--metadata", help="New metadata (JSON object)")
        parser.add_argument("--from-file", help="Load metadata from file")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return
        except argparse.ArgumentError as e:
            logger.error(f"Argument parsing error: {e}")
            print(f"Error: {e}")
            return

        # Get vector ID
        vector_id = parsed_args.id

        # Get metadata
        metadata = None

        if parsed_args.metadata:
            # Parse JSON metadata
            try:
                metadata = json.loads(parsed_args.metadata)
            except json.JSONDecodeError as e:
                print(f"Error parsing metadata: {e}")
                return

        elif parsed_args.from_file:
            # Load metadata from file
            try:
                with open(parsed_args.from_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata from file: {e}")
                return

        else:
            print("No metadata provided")
            return

        # Update metadata
        try:
            success = await self.wdbx.update_metadata_async(vector_id, metadata)
            if success:
                print(f"Metadata updated for vector: {vector_id}")
            else:
                print(f"Vector not found: {vector_id}")
        except Exception as e:
            print(f"Error updating metadata: {e}")

    async def _cmd_stats(self, args: str):
        """Command handler for the stats command."""
        try:
            stats = self.wdbx.get_stats()
            print("Database statistics:")

            # Print formatted stats
            for key, value in stats.items():
                if key == "plugins_loaded":
                    print(f"  Plugins loaded: {value}")
                    if value > 0 and "plugins" in stats:
                        plugins = [plugin["name"]
                                   for plugin in stats["plugins"]]
                        print(f"    {', '.join(plugins)}")
                elif key == "indices":
                    print(f"  Indices: {len(value)}")
                    for i, index in enumerate(value):
                        print(
                            f"    Shard {i}: {index['type']}, {index['size']} vectors")
                elif key != "plugins":  # Skip plugins, handled above
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error getting statistics: {e}")

    async def _cmd_clear(self, args: str):
        """Command handler for the clear command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(description="Clear the database")
        parser.add_argument("--force", action="store_true",
                            help="Skip confirmation prompt")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        # Confirm unless forced
        if not parsed_args.force:
            confirm = input(
                "Are you sure you want to clear the database? (y/N) ")
            if confirm.lower() != "y":
                print("Operation cancelled.")
                return

        # Clear database
        try:
            count = await self.wdbx.clear_async()
            print(f"Database cleared: {count} vectors removed")
        except Exception as e:
            print(f"Error clearing database: {e}")

    async def _cmd_plugins(self, args: str):
        """Command handler for the plugins command."""
        try:
            if not self.wdbx.plugins:
                print("No plugins loaded")
                return

            print(f"Loaded plugins ({len(self.wdbx.plugins)}):")
            for plugin_name, plugin in self.wdbx.plugins.items():
                print(f"  {plugin_name} v{plugin.version}: {plugin.description}")
        except Exception as e:
            print(f"Error listing plugins: {e}")

    async def _cmd_plugin_info(self, args: str):
        """Command handler for the plugin-info command."""
        if not args:
            print("Plugin name required")
            return

        plugin_name = args.strip()

        try:
            plugin = self.wdbx.get_plugin(plugin_name)
            if plugin is None:
                print(f"Plugin not found: {plugin_name}")
                return

            print(f"Plugin: {plugin.name} v{plugin.version}")
            print(f"Description: {plugin.description}")

            # Get plugin stats
            try:
                stats = plugin.get_stats()
                if stats:
                    print("Statistics:")
                    for key, value in stats.items():
                        # Skip fields already shown above
                        if key not in ["name", "version", "description"]:
                            print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error getting plugin stats: {e}")

            # Get plugin help
            if hasattr(plugin, "get_help"):
                try:
                    help_text = plugin.get_help()
                    print("\nHelp:")
                    print(help_text)
                except Exception as e:
                    print(f"Error getting plugin help: {e}")
        except Exception as e:
            print(f"Error getting plugin info: {e}")

    async def _cmd_serve(self, args: str):
        """Command handler for the serve command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(description="Start the API server")
        parser.add_argument("--host", default="0.0.0.0",
                            help="Host to bind to")
        parser.add_argument("--port", type=int, default=8000,
                            help="Port to listen on")
        parser.add_argument("--auth", action="store_true",
                            help="Enable authentication")
        parser.add_argument("--auth-key", help="Authentication key")
        parser.add_argument("--cors", action="store_true", help="Enable CORS")
        parser.add_argument(
            "--cors-origins", help="Allowed CORS origins (comma-separated)")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        # Import API server
        try:
            from .api.server import WDBXAPIServer
        except ImportError as e:
            print(f"API server not available: {e}")
            print("Install required packages with: pip install fastapi uvicorn")
            return

        # Prepare CORS origins
        cors_origins = None
        if parsed_args.cors_origins:
            cors_origins = parsed_args.cors_origins.split(",")

        # Create and start API server
        try:
            server = WDBXAPIServer(
                self.wdbx,
                host=parsed_args.host,
                port=parsed_args.port,
                enable_auth=parsed_args.auth,
                auth_key=parsed_args.auth_key,
                enable_cors=parsed_args.cors,
                cors_origins=cors_origins,
            )

            print(
                f"Starting API server on {parsed_args.host}:{parsed_args.port}")
            print("Press Ctrl+C to stop the server")

            await server.initialize()
            await server.start()
        except Exception as e:
            print(f"Error starting API server: {e}")


def main():
    """CLI entry point."""
    # Create WDBX instance
    wdbx = WDBX(
        vector_dimension=384,
        enable_plugins=True,
    )

    # Create CLI
    cli = WDBXCLI(wdbx)

    # Run CLI
    try:
        loop = asyncio.get_event_loop()

        # Initialize WDBX
        loop.run_until_complete(wdbx.initialize())

        # Run CLI
        loop.run_until_complete(cli.run(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Shut down WDBX
        try:
            loop.run_until_complete(wdbx.shutdown())
        except Exception as e:
            print(f"Error shutting down: {e}")


if __name__ == "__main__":
    main()
