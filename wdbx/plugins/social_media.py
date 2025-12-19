"""
Social media integration plugin for WDBX.

This plugin provides unified access to various social media platforms.
"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import aiohttp

from ..plugins.base import WDBXPlugin, PluginError

logger = logging.getLogger(__name__)


class SocialMediaPlugin(WDBXPlugin):
    """
    Social media integration plugin for WDBX.

    This plugin provides unified access to various social media platforms,
    including Twitter, Reddit, and others.

    Attributes:
        wdbx: Reference to the WDBX instance
        enabled_platforms: List of enabled platform names
        platform_clients: Dictionary mapping platform names to client instances
    """

    def __init__(self, wdbx):
        """
        Initialize the social media plugin.

        Args:
            wdbx: Reference to the WDBX instance
        """
        super().__init__(wdbx)

        # Load configuration
        platforms_config = self.get_config("ENABLED_PLATFORMS", "twitter,reddit")
        self.enabled_platforms = [
            p.strip() for p in platforms_config.split(",") if p.strip()
        ]
        self.cache_ttl = int(self.get_config("CACHE_TTL", 300))  # 5 minutes
        self.demo_mode = self.get_config("DEMO_MODE", False)

        # Platform-specific configuration
        self.config = {
            "twitter": {
                "api_key": self.get_config("TWITTER_API_KEY", ""),
                "api_secret": self.get_config("TWITTER_API_SECRET", ""),
                "access_token": self.get_config("TWITTER_ACCESS_TOKEN", ""),
                "access_secret": self.get_config("TWITTER_ACCESS_SECRET", ""),
                "bearer_token": self.get_config("TWITTER_BEARER_TOKEN", ""),
            },
            "reddit": {
                "client_id": self.get_config("REDDIT_CLIENT_ID", ""),
                "client_secret": self.get_config("REDDIT_CLIENT_SECRET", ""),
                "user_agent": self.get_config(
                    "REDDIT_USER_AGENT", f"WDBX:SocialMediaPlugin:v{self.version}"
                ),
            },
            "facebook": {
                "app_id": self.get_config("FACEBOOK_APP_ID", ""),
                "app_secret": self.get_config("FACEBOOK_APP_SECRET", ""),
                "access_token": self.get_config("FACEBOOK_ACCESS_TOKEN", ""),
            },
        }

        # Platform clients
        self.platform_clients = {}

        # Cache
        self.cache = {}
        self.cache_timestamps = {}

        # HTTP session
        self.session = None

        logger.info(
            f"Initialized SocialMediaPlugin with platforms: {', '.join(self.enabled_platforms)}"
        )

    @property
    def name(self) -> str:
        """Return the name of the plugin."""
        return "socialmedia"

    @property
    def description(self) -> str:
        """Return a description of the plugin."""
        return "Social media integration plugin for WDBX, providing unified access to various platforms."

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "0.2.0"

    async def initialize(self) -> None:
        """Initialize the plugin."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()

            # Initialize platform clients
            await self._initialize_platforms()

            logger.info(f"SocialMediaPlugin initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SocialMediaPlugin: {e}")
            raise PluginError(f"Error initializing SocialMediaPlugin: {e}")

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        # Close platform clients
        for platform, client in self.platform_clients.items():
            if hasattr(client, "close") and callable(client.close):
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing {platform} client: {e}")

        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None

        logger.info("SocialMediaPlugin shut down")

    async def _initialize_platforms(self) -> None:
        """Initialize platform clients."""
        for platform in self.enabled_platforms:
            try:
                if platform == "twitter":
                    await self._initialize_twitter()
                elif platform == "reddit":
                    await self._initialize_reddit()
                elif platform == "facebook":
                    await self._initialize_facebook()
                else:
                    logger.warning(f"Unsupported platform: {platform}")
            except Exception as e:
                logger.error(f"Error initializing {platform} client: {e}")

    async def _initialize_twitter(self) -> None:
        """Initialize Twitter client."""
        # Skip in demo mode
        if self.demo_mode:
            self.platform_clients["twitter"] = "DEMO"
            logger.info("Twitter client initialized in demo mode")
            return

        try:
            import tweepy

            # Check for required credentials
            if (
                self.config["twitter"]["api_key"]
                and self.config["twitter"]["api_secret"]
                and self.config["twitter"]["access_token"]
                and self.config["twitter"]["access_secret"]
            ):
                # Create client with OAuth 1.0a
                client = tweepy.Client(
                    consumer_key=self.config["twitter"]["api_key"],
                    consumer_secret=self.config["twitter"]["api_secret"],
                    access_token=self.config["twitter"]["access_token"],
                    access_token_secret=self.config["twitter"]["access_secret"],
                )

                self.platform_clients["twitter"] = client
                logger.info("Twitter client initialized with OAuth 1.0a")

            elif self.config["twitter"]["bearer_token"]:
                # Create client with bearer token
                client = tweepy.Client(
                    bearer_token=self.config["twitter"]["bearer_token"]
                )

                self.platform_clients["twitter"] = client
                logger.info("Twitter client initialized with bearer token")

            else:
                raise PluginError("Missing Twitter credentials")

        except ImportError:
            logger.error("tweepy not installed, required for Twitter integration")
            raise PluginError(
                "tweepy is required for Twitter integration. Install with: pip install tweepy"
            )

    async def _initialize_reddit(self) -> None:
        """Initialize Reddit client."""
        # Skip in demo mode
        if self.demo_mode:
            self.platform_clients["reddit"] = "DEMO"
            logger.info("Reddit client initialized in demo mode")
            return

        try:
            import praw

            # Check for required credentials
            if (
                self.config["reddit"]["client_id"]
                and self.config["reddit"]["client_secret"]
            ):
                # Create client
                client = praw.Reddit(
                    client_id=self.config["reddit"]["client_id"],
                    client_secret=self.config["reddit"]["client_secret"],
                    user_agent=self.config["reddit"]["user_agent"],
                )

                self.platform_clients["reddit"] = client
                logger.info("Reddit client initialized")
            else:
                raise PluginError("Missing Reddit credentials")

        except ImportError:
            logger.error("praw not installed, required for Reddit integration")
            raise PluginError(
                "praw is required for Reddit integration. Install with: pip install praw"
            )

    async def _initialize_facebook(self) -> None:
        """Initialize Facebook client."""
        # Skip in demo mode
        if self.demo_mode:
            self.platform_clients["facebook"] = "DEMO"
            logger.info("Facebook client initialized in demo mode")
            return

        try:
            import facebook

            # Check for required credentials
            if self.config["facebook"]["access_token"]:
                # Create client
                client = facebook.GraphAPI(
                    access_token=self.config["facebook"]["access_token"]
                )

                self.platform_clients["facebook"] = client
                logger.info("Facebook client initialized")
            else:
                raise PluginError("Missing Facebook credentials")

        except ImportError:
            logger.error(
                "facebook-sdk not installed, required for Facebook integration"
            )
            raise PluginError(
                "facebook-sdk is required for Facebook integration. Install with: pip install facebook-sdk"
            )

    def _cache_key(self, platform: str, method: str, *args, **kwargs) -> str:
        """
        Generate a cache key for a method call.

        Args:
            platform: Platform name
            method: Method name
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Cache key string
        """
        # Convert args and kwargs to strings for key generation
        arg_str = ",".join(str(arg) for arg in args)
        kwarg_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

        return f"{platform}:{method}:{arg_str}:{kwarg_str}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache if it exists and is not expired.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key in self.cache and key in self.cache_timestamps:
            timestamp = self.cache_timestamps[key]
            if time.time() - timestamp <= self.cache_ttl:
                return self.cache[key]

        return None

    def _set_in_cache(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store
        """
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

    async def search_posts(
        self,
        query: str,
        platforms: Optional[List[str]] = None,
        limit: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for posts across platforms.

        Args:
            query: Search query
            platforms: List of platforms to search (default: all enabled platforms)
            limit: Maximum number of results per platform
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary mapping platform names to lists of post dictionaries

        Raises:
            PluginError: If search fails
        """
        # Use all enabled platforms if not specified
        if platforms is None:
            platforms = self.enabled_platforms

        # Filter only enabled platforms
        platforms = [p for p in platforms if p in self.enabled_platforms]

        if not platforms:
            raise PluginError("No enabled platforms specified")

        # Generate cache key
        cache_key = self._cache_key(
            "search_posts", query, platforms=platforms, limit=limit
        )

        # Check cache
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

        # Search on all platforms
        results = {}
        tasks = []

        for platform in platforms:
            if platform == "twitter":
                tasks.append(self._search_twitter(query, limit))
            elif platform == "reddit":
                tasks.append(self._search_reddit(query, limit))
            elif platform == "facebook":
                tasks.append(self._search_facebook(query, limit))

        # Wait for all tasks to complete
        platform_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for platform, result in zip(platforms, platform_results):
            if isinstance(result, Exception):
                logger.error(f"Error searching on {platform}: {result}")
                results[platform] = {"error": str(result)}
            else:
                results[platform] = result

        # Cache results
        self._set_in_cache(cache_key, results)

        return results

    async def _search_twitter(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search for posts on Twitter.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of post dictionaries

        Raises:
            PluginError: If search fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("twitter") == "DEMO":
            # Return mock data
            return [
                {
                    "id": f"tweet_{i}",
                    "text": f"Demo tweet about {query} #{i+1}",
                    "user": {"screen_name": f"user_{i}", "name": f"User {i}"},
                    "created_at": "2023-01-01T00:00:00Z",
                    "platform": "twitter",
                }
                for i in range(min(limit, 5))
            ]

        # Check if client is available
        client = self.platform_clients.get("twitter")
        if not client:
            raise PluginError("Twitter client not initialized")

        try:
            import tweepy

            # Search for tweets
            response = client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                tweet_fields=["created_at", "public_metrics", "entities"],
                user_fields=["name", "username", "profile_image_url"],
                expansions=["author_id"],
            )

            # Process results
            results = []
            users = (
                {user.id: user for user in response.includes.get("users", [])}
                if response.includes
                else {}
            )

            for tweet in response.data or []:
                user = users.get(tweet.author_id, None)
                tweet_data = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": (
                        tweet.created_at.isoformat()
                        if hasattr(tweet, "created_at")
                        else None
                    ),
                    "metrics": (
                        tweet.public_metrics
                        if hasattr(tweet, "public_metrics")
                        else None
                    ),
                    "platform": "twitter",
                }

                if user:
                    tweet_data["user"] = {
                        "id": user.id,
                        "name": user.name,
                        "username": user.username,
                        "profile_image_url": (
                            user.profile_image_url
                            if hasattr(user, "profile_image_url")
                            else None
                        ),
                    }

                results.append(tweet_data)

            return results

        except Exception as e:
            logger.error(f"Error searching Twitter: {e}")
            raise PluginError(f"Error searching Twitter: {e}")

    async def _search_reddit(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search for posts on Reddit.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of post dictionaries

        Raises:
            PluginError: If search fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("reddit") == "DEMO":
            # Return mock data
            return [
                {
                    "id": f"reddit_{i}",
                    "title": f"Demo Reddit post about {query} #{i+1}",
                    "text": f"This is a mock Reddit post about {query} for demonstration purposes.",
                    "author": f"reddit_user_{i}",
                    "subreddit": f"r/demo",
                    "score": 100 - i * 10,
                    "created_at": "2023-01-01T00:00:00Z",
                    "platform": "reddit",
                }
                for i in range(min(limit, 5))
            ]

        # Check if client is available
        client = self.platform_clients.get("reddit")
        if not client:
            raise PluginError("Reddit client not initialized")

        try:
            # Search for posts
            submissions = client.subreddit("all").search(query, limit=limit)

            # Process results
            results = []

            for submission in submissions:
                # Convert to dict to avoid praw object serialization issues
                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "text": (
                        submission.selftext if hasattr(submission, "selftext") else ""
                    ),
                    "url": submission.url if hasattr(submission, "url") else None,
                    "author": (
                        submission.author.name if submission.author else "[deleted]"
                    ),
                    "subreddit": (
                        submission.subreddit.display_name
                        if hasattr(submission, "subreddit")
                        else None
                    ),
                    "score": submission.score,
                    "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(submission.created_utc)
                    ),
                    "num_comments": (
                        submission.num_comments
                        if hasattr(submission, "num_comments")
                        else 0
                    ),
                    "platform": "reddit",
                }

                results.append(post_data)

            return results

        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            raise PluginError(f"Error searching Reddit: {e}")

    async def _search_facebook(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search for posts on Facebook.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of post dictionaries

        Raises:
            PluginError: If search fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("facebook") == "DEMO":
            # Return mock data
            return [
                {
                    "id": f"fb_{i}",
                    "message": f"Demo Facebook post about {query} #{i+1}",
                    "from": {"name": f"Facebook User {i}", "id": f"fb_user_{i}"},
                    "created_time": "2023-01-01T00:00:00Z",
                    "platform": "facebook",
                }
                for i in range(min(limit, 5))
            ]

        # Check if client is available
        client = self.platform_clients.get("facebook")
        if not client:
            raise PluginError("Facebook client not initialized")

        try:
            # Note: Facebook API has limited search capabilities
            # This is a simplified implementation
            # Graph API requires specific permissions for search

            # For demonstration purposes, we'll search public posts
            # This requires pages_read_engagement permission
            response = client.get_object(
                id="search",
                fields="id,message,created_time,from",
                limit=limit,
                q=query,
                type="post",
            )

            # Process results
            results = []

            for post in response.get("data", []):
                post_data = {
                    "id": post.get("id"),
                    "message": post.get("message", ""),
                    "created_time": post.get("created_time"),
                    "from": post.get("from"),
                    "platform": "facebook",
                }

                results.append(post_data)

            return results

        except Exception as e:
            logger.error(f"Error searching Facebook: {e}")
            raise PluginError(f"Error searching Facebook: {e}")

    async def get_user_profile(
        self,
        username: str,
        platform: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Get a user profile from a platform.

        Args:
            username: Username to look up
            platform: Platform name
            use_cache: Whether to use cached results if available

        Returns:
            User profile dictionary

        Raises:
            PluginError: If profile retrieval fails
        """
        # Check if platform is enabled
        if platform not in self.enabled_platforms:
            raise PluginError(f"Platform not enabled: {platform}")

        # Generate cache key
        cache_key = self._cache_key("get_user_profile", username, platform=platform)

        # Check cache
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

        # Get profile from platform
        result = None

        if platform == "twitter":
            result = await self._get_twitter_profile(username)
        elif platform == "reddit":
            result = await self._get_reddit_profile(username)
        elif platform == "facebook":
            result = await self._get_facebook_profile(username)
        else:
            raise PluginError(f"Unsupported platform: {platform}")

        # Cache result
        self._set_in_cache(cache_key, result)

        return result

    async def _get_twitter_profile(self, username: str) -> Dict[str, Any]:
        """
        Get a user profile from Twitter.

        Args:
            username: Twitter username (without @)

        Returns:
            User profile dictionary

        Raises:
            PluginError: If profile retrieval fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("twitter") == "DEMO":
            # Return mock data
            return {
                "id": f"twitter_user_123",
                "name": f"Demo Twitter User",
                "username": username,
                "description": f"This is a demo Twitter profile for {username}",
                "followers_count": 1000,
                "following_count": 500,
                "tweet_count": 5000,
                "profile_image_url": "https://example.com/profile.jpg",
                "platform": "twitter",
            }

        # Check if client is available
        client = self.platform_clients.get("twitter")
        if not client:
            raise PluginError("Twitter client not initialized")

        try:
            # Get user profile
            response = client.get_user(
                username=username,
                user_fields=[
                    "description",
                    "public_metrics",
                    "profile_image_url",
                    "created_at",
                ],
            )

            if not response.data:
                raise PluginError(f"Twitter user not found: {username}")

            user = response.data

            # Process user data
            user_data = {
                "id": user.id,
                "name": user.name,
                "username": user.username,
                "description": (
                    user.description if hasattr(user, "description") else None
                ),
                "platform": "twitter",
            }

            # Add metrics if available
            if hasattr(user, "public_metrics"):
                metrics = user.public_metrics
                user_data.update(
                    {
                        "followers_count": metrics.get("followers_count"),
                        "following_count": metrics.get("following_count"),
                        "tweet_count": metrics.get("tweet_count"),
                    }
                )

            # Add profile image if available
            if hasattr(user, "profile_image_url"):
                user_data["profile_image_url"] = user.profile_image_url

            # Add creation date if available
            if hasattr(user, "created_at"):
                user_data["created_at"] = user.created_at.isoformat()

            return user_data

        except Exception as e:
            logger.error(f"Error getting Twitter profile: {e}")
            raise PluginError(f"Error getting Twitter profile: {e}")

    async def _get_reddit_profile(self, username: str) -> Dict[str, Any]:
        """
        Get a user profile from Reddit.

        Args:
            username: Reddit username (without u/)

        Returns:
            User profile dictionary

        Raises:
            PluginError: If profile retrieval fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("reddit") == "DEMO":
            # Return mock data
            return {
                "id": f"reddit_user_123",
                "name": username,
                "created_at": "2023-01-01T00:00:00Z",
                "comment_karma": 5000,
                "link_karma": 2000,
                "is_gold": False,
                "platform": "reddit",
            }

        # Check if client is available
        client = self.platform_clients.get("reddit")
        if not client:
            raise PluginError("Reddit client not initialized")

        try:
            # Get user profile
            user = client.redditor(username)

            # Trigger a request to check if user exists
            _ = user.created_utc

            # Process user data
            user_data = {
                "id": user.id if hasattr(user, "id") else username,
                "name": user.name,
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(user.created_utc)
                ),
                "comment_karma": user.comment_karma,
                "link_karma": user.link_karma,
                "is_gold": user.is_gold if hasattr(user, "is_gold") else False,
                "platform": "reddit",
            }

            return user_data

        except Exception as e:
            logger.error(f"Error getting Reddit profile: {e}")
            raise PluginError(f"Error getting Reddit profile: {e}")

    async def _get_facebook_profile(self, username: str) -> Dict[str, Any]:
        """
        Get a user profile from Facebook.

        Args:
            username: Facebook username or ID

        Returns:
            User profile dictionary

        Raises:
            PluginError: If profile retrieval fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("facebook") == "DEMO":
            # Return mock data
            return {
                "id": f"fb_user_123",
                "name": f"Demo Facebook User",
                "username": username,
                "platform": "facebook",
            }

        # Check if client is available
        client = self.platform_clients.get("facebook")
        if not client:
            raise PluginError("Facebook client not initialized")

        try:
            # Note: Facebook API requires specific permissions for user data
            # This is a simplified implementation
            user = client.get_object(username, fields="id,name,username,about,picture")

            # Process user data
            user_data = {
                "id": user.get("id"),
                "name": user.get("name"),
                "username": user.get("username"),
                "about": user.get("about"),
                "platform": "facebook",
            }

            # Add profile picture if available
            if "picture" in user and "data" in user["picture"]:
                user_data["picture_url"] = user["picture"]["data"].get("url")

            return user_data

        except Exception as e:
            logger.error(f"Error getting Facebook profile: {e}")
            raise PluginError(f"Error getting Facebook profile: {e}")

    async def get_trending_topics(
        self,
        platform: str,
        location: Optional[str] = None,
        limit: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics from a platform.

        Args:
            platform: Platform name
            location: Optional location for localized trends
            limit: Maximum number of trends to return
            use_cache: Whether to use cached results if available

        Returns:
            List of trending topic dictionaries

        Raises:
            PluginError: If trending topic retrieval fails
        """
        # Check if platform is enabled
        if platform not in self.enabled_platforms:
            raise PluginError(f"Platform not enabled: {platform}")

        # Generate cache key
        cache_key = self._cache_key(
            "get_trending_topics", platform=platform, location=location, limit=limit
        )

        # Check cache
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

        # Get trending topics from platform
        result = None

        if platform == "twitter":
            result = await self._get_twitter_trends(location, limit)
        elif platform == "reddit":
            result = await self._get_reddit_trends(limit)
        elif platform == "facebook":
            result = await self._get_facebook_trends(limit)
        else:
            raise PluginError(f"Unsupported platform: {platform}")

        # Cache result
        self._set_in_cache(cache_key, result)

        return result

    async def _get_twitter_trends(
        self, location: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics from Twitter.

        Args:
            location: Optional location name or WOEID
            limit: Maximum number of trends to return

        Returns:
            List of trending topic dictionaries

        Raises:
            PluginError: If trend retrieval fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("twitter") == "DEMO":
            # Return mock data
            return [
                {
                    "name": f"#Trending{i}",
                    "query": f"%23Trending{i}",
                    "tweet_volume": 10000 - i * 1000,
                    "platform": "twitter",
                }
                for i in range(min(limit, 10))
            ]

        # Check if client is available
        # For Twitter trends we need a v1 API client
        try:
            import tweepy

            # Create v1 API client if not already available
            if "twitter_v1" not in self.platform_clients:
                # Check for OAuth credentials
                if (
                    self.config["twitter"]["api_key"]
                    and self.config["twitter"]["api_secret"]
                    and self.config["twitter"]["access_token"]
                    and self.config["twitter"]["access_secret"]
                ):

                    auth = tweepy.OAuth1UserHandler(
                        self.config["twitter"]["api_key"],
                        self.config["twitter"]["api_secret"],
                        self.config["twitter"]["access_token"],
                        self.config["twitter"]["access_secret"],
                    )

                    api = tweepy.API(auth)
                    self.platform_clients["twitter_v1"] = api
                else:
                    raise PluginError(
                        "Twitter OAuth 1.0a credentials required for trends"
                    )

            api = self.platform_clients["twitter_v1"]

            # Get WOEID for location
            woeid = 1  # Default to worldwide

            if location:
                try:
                    # Try to parse as WOEID (integer)
                    woeid = int(location)
                except ValueError:
                    # Try to find WOEID by location name
                    available_locations = api.trends_available()
                    for loc in available_locations:
                        if location.lower() in loc["name"].lower():
                            woeid = loc["woeid"]
                            break
                    else:
                        raise PluginError(f"Location not found: {location}")

            # Get trends
            trends = api.trends_place(woeid)[0]["trends"]

            # Process trends
            results = []

            for trend in trends[:limit]:
                trend_data = {
                    "name": trend["name"],
                    "query": trend["query"],
                    "tweet_volume": trend["tweet_volume"],
                    "platform": "twitter",
                }

                results.append(trend_data)

            return results

        except Exception as e:
            logger.error(f"Error getting Twitter trends: {e}")
            raise PluginError(f"Error getting Twitter trends: {e}")

    async def _get_reddit_trends(self, limit: int) -> List[Dict[str, Any]]:
        """
        Get trending topics from Reddit.

        Args:
            limit: Maximum number of trends to return

        Returns:
            List of trending topic dictionaries

        Raises:
            PluginError: If trend retrieval fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("reddit") == "DEMO":
            # Return mock data
            return [
                {
                    "title": f"Trending Reddit Topic {i}",
                    "subreddit": f"r/popular",
                    "score": 5000 - i * 300,
                    "platform": "reddit",
                }
                for i in range(min(limit, 10))
            ]

        # Check if client is available
        client = self.platform_clients.get("reddit")
        if not client:
            raise PluginError("Reddit client not initialized")

        try:
            # Get hot posts from r/popular or r/all as trends
            submissions = client.subreddit("popular").hot(limit=limit)

            # Process submissions
            results = []

            for submission in submissions:
                trend_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "subreddit": (
                        submission.subreddit.display_name
                        if hasattr(submission, "subreddit")
                        else None
                    ),
                    "score": submission.score,
                    "num_comments": (
                        submission.num_comments
                        if hasattr(submission, "num_comments")
                        else 0
                    ),
                    "platform": "reddit",
                }

                results.append(trend_data)

            return results

        except Exception as e:
            logger.error(f"Error getting Reddit trends: {e}")
            raise PluginError(f"Error getting Reddit trends: {e}")

    async def _get_facebook_trends(self, limit: int) -> List[Dict[str, Any]]:
        """
        Get trending topics from Facebook.

        Args:
            limit: Maximum number of trends to return

        Returns:
            List of trending topic dictionaries

        Raises:
            PluginError: If trend retrieval fails
        """
        # Demo mode
        if self.demo_mode or self.platform_clients.get("facebook") == "DEMO":
            # Return mock data
            return [
                {
                    "name": f"Trending Facebook Topic {i}",
                    "platform": "facebook",
                }
                for i in range(min(limit, 10))
            ]

        # Facebook no longer provides a public trends API
        # This is a placeholder that returns demo data
        return [
            {
                "name": f"Trending Facebook Topic {i}",
                "platform": "facebook",
            }
            for i in range(min(limit, 10))
        ]

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for social media content.

        Args:
            text: The input text

        Returns:
            Embedding vector as a list of floats

        Raises:
            PluginError: If embedding creation fails
        """
        # Try to find an embedding plugin to delegate to
        for plugin_name in [
            "openai",
            "ollama",
            "lmstudio",
            "huggingface",
            "sentencetransformers",
        ]:
            if plugin_name in self.wdbx.plugins:
                plugin = self.wdbx.plugins[plugin_name]
                try:
                    embedding = await plugin.create_embedding(text)
                    return embedding
                except Exception as e:
                    logger.error(f"Error creating embedding with {plugin_name}: {e}")

        # If no embedding plugin is available, try to use SentenceTransformers locally
        try:
            from sentence_transformers import SentenceTransformer

            # Load model
            model_name = "all-MiniLM-L6-v2"  # Small, fast model
            model = SentenceTransformer(model_name)

            # Generate embedding
            embedding = model.encode(text)

            return embedding.tolist()
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise PluginError(
                "No embedding plugin available and sentence-transformers not installed"
            )
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise PluginError(f"Error creating embedding: {e}")

    def register_commands(self) -> None:
        """Register commands with the WDBX CLI."""
        if hasattr(self.wdbx, "register_command"):
            self.wdbx.register_command(
                "socialmedia-search",
                self._cmd_search,
                "Search for posts across social media platforms",
                {
                    "--query": "Search query",
                    "--platform": "Platform to search (default: all enabled)",
                    "--limit": "Maximum number of results per platform",
                },
            )

            self.wdbx.register_command(
                "socialmedia-profile",
                self._cmd_profile,
                "Get a user profile from a social media platform",
                {
                    "--username": "Username to look up",
                    "--platform": "Platform name",
                },
            )

            self.wdbx.register_command(
                "socialmedia-trending",
                self._cmd_trending,
                "Get trending topics from a social media platform",
                {
                    "--platform": "Platform name",
                    "--location": "Optional location for localized trends",
                    "--limit": "Maximum number of trends",
                },
            )

            self.wdbx.register_command(
                "socialmedia-platforms",
                self._cmd_platforms,
                "List enabled social media platforms",
                {},
            )

    async def _cmd_search(self, args: str):
        """Command handler for the socialmedia-search command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Search for posts across social media platforms"
        )
        parser.add_argument("--query", required=True, help="Search query")
        parser.add_argument(
            "--platform", help="Platform to search (default: all enabled)"
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=5,
            help="Maximum number of results per platform",
        )

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Determine platforms
            platforms = None
            if parsed_args.platform:
                platforms = [parsed_args.platform]

            # Search for posts
            results = await self.search_posts(
                query=parsed_args.query, platforms=platforms, limit=parsed_args.limit
            )

            # Print results
            for platform, posts in results.items():
                print(f"\n=== Results from {platform.capitalize()} ===")

                if isinstance(posts, dict) and "error" in posts:
                    print(f"Error: {posts['error']}")
                    continue

                if not posts:
                    print(f"No results found")
                    continue

                for i, post in enumerate(posts):
                    print(f"\n{i+1}. ", end="")

                    # Print post details based on platform
                    if platform == "twitter":
                        user = post.get("user", {})
                        print(
                            f"@{user.get('username', 'unknown')}: {post.get('text', '')}"
                        )
                    elif platform == "reddit":
                        print(
                            f"r/{post.get('subreddit', 'unknown')} - {post.get('title', '')}"
                        )
                        if "text" in post and post["text"]:
                            text = post["text"]
                            if len(text) > 100:
                                text = text[:100] + "..."
                            print(f"  {text}")
                    elif platform == "facebook":
                        user = post.get("from", {})
                        print(
                            f"{user.get('name', 'unknown')}: {post.get('message', '')}"
                        )
                    else:
                        print(json.dumps(post, indent=2))
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_profile(self, args: str):
        """Command handler for the socialmedia-profile command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Get a user profile from a social media platform"
        )
        parser.add_argument("--username", required=True, help="Username to look up")
        parser.add_argument("--platform", required=True, help="Platform name")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Get user profile
            profile = await self.get_user_profile(
                username=parsed_args.username, platform=parsed_args.platform
            )

            # Print profile
            print(f"\n=== {parsed_args.platform.capitalize()} Profile ===")
            print(json.dumps(profile, indent=2))
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_trending(self, args: str):
        """Command handler for the socialmedia-trending command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Get trending topics from a social media platform"
        )
        parser.add_argument("--platform", required=True, help="Platform name")
        parser.add_argument("--location", help="Optional location for localized trends")
        parser.add_argument(
            "--limit", type=int, default=10, help="Maximum number of trends"
        )

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Get trending topics
            trends = await self.get_trending_topics(
                platform=parsed_args.platform,
                location=parsed_args.location,
                limit=parsed_args.limit,
            )

            # Print trends
            location_str = (
                f" in {
                parsed_args.location} "
                if parsed_args.location
                else ""
            )
            print(
                f"\n=== Trending on {parsed_args.platform.capitalize()}{location_str} ==="
            )

            for i, trend in enumerate(trends):
                print(f"\n{i+1}. ", end="")

                # Print trend details based on platform
                if parsed_args.platform == "twitter":
                    volume = trend.get("tweet_volume", "N/A")
                    print(f"{trend.get('name', '')} ({volume} tweets)")
                elif parsed_args.platform == "reddit":
                    print(f"{trend.get('title', '')} - r/{trend.get('subreddit', '')}")
                    print(
                        f"  Score: {trend.get('score', 'N/A')}, Comments: {trend.get('num_comments', 'N/A')}"
                    )
                else:
                    print(trend.get("name", ""))
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_platforms(self, args: str):
        """Command handler for the socialmedia-platforms command."""
        try:
            # Print enabled platforms
            print("\n=== Enabled Social Media Platforms ===")

            if not self.enabled_platforms:
                print("No platforms enabled")
                return

            for platform in self.enabled_platforms:
                client = self.platform_clients.get(platform)
                status = "Connected"

                if client is None:
                    status = "Not initialized"
                elif client == "DEMO":
                    status = "Demo mode"

                print(f"- {platform.capitalize()}: {status}")

            # Print demo mode status
            if self.demo_mode:
                print("\nRunning in demo mode (no API credentials required)")
        except Exception as e:
            print(f"Error: {e}")
