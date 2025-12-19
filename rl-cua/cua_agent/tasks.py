"""Sample tasks for CUA Agent GRPO training.

This module defines 10 sample Android tasks for training and evaluation.
Each task can be validated via gbox system state APIs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskCategory(str, Enum):
    """Task categories."""
    SYSTEM = "system"
    NAVIGATION = "navigation"
    SETTINGS = "settings"
    APP = "app"
    INPUT = "input"


@dataclass
class CUATask:
    """A task for CUA agent to complete on Android."""
    
    id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    category: TaskCategory
    max_steps: int = 10
    
    # Validation config
    validation_type: str = "state"  # "state", "screenshot", "api"
    validation_query: Optional[str] = None  # Query to check against gbox state
    expected_result: Optional[Any] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "category": self.category.value,
            "max_steps": self.max_steps,
            "validation_type": self.validation_type,
            "validation_query": self.validation_query,
            "expected_result": self.expected_result,
            "tags": self.tags,
            "prerequisites": self.prerequisites,
        }

# =============================================================================
# TRAINING TASKS (80 tasks)
# =============================================================================

TRAINING_TASKS = [
    # Task 1: Open Settings app
    CUATask(
        id="train_01_open_settings",
        name="Open Settings",
        description="Open the Settings app from the home screen.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.settings",
        tags=["app", "settings", "launch"],
    ),

    # Task 2: Enable WiFi
    CUATask(
        id="train_02_enable_wifi",
        name="Enable WiFi",
        description="Go to Settings and turn on WiFi if it's off.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="wifi_enabled",
        expected_result=True,
        tags=["settings", "wifi", "toggle"],
    ),

    # Task 3: Set screen brightness to maximum
    CUATask(
        id="train_03_max_brightness",
        name="Maximum Brightness",
        description="Open Settings and set the screen brightness to maximum.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="brightness_level",
        expected_result=255,
        tags=["settings", "display", "brightness"],
    ),

    # Task 4: Open Chrome browser
    CUATask(
        id="train_04_open_chrome",
        name="Open Chrome",
        description="Open the Chrome browser app.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.chrome",
        tags=["app", "browser", "chrome"],
    ),

    # Task 5: Enable Airplane Mode
    CUATask(
        id="train_05_airplane_mode",
        name="Enable Airplane Mode",
        description="Go to Settings and turn on Airplane Mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="airplane_mode",
        expected_result=True,
        tags=["settings", "network", "airplane"],
    ),

    # Task 6: Check battery level (open Battery page)
    CUATask(
        id="train_06_check_battery",
        name="Check Battery Level",
        description="Open Settings and navigate to the Battery section to check the current battery level.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="current_activity",
        expected_result="battery",
        tags=["settings", "battery", "info"],
    ),

    # Task 7: Go to home screen
    CUATask(
        id="train_07_go_home",
        name="Go to Home Screen",
        description="Press the home button to return to the home screen.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.NAVIGATION,
        max_steps=3,
        validation_type="state",
        validation_query="is_home_screen",
        expected_result=True,
        tags=["navigation", "home"],
    ),

    # Task 8: Enable Bluetooth
    CUATask(
        id="train_08_enable_bluetooth",
        name="Enable Bluetooth",
        description="Go to Settings and turn on Bluetooth if it's off.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="bluetooth_enabled",
        expected_result=True,
        tags=["settings", "bluetooth", "toggle"],
    ),

    # Task 9: Disable WiFi
    CUATask(
        id="train_09_disable_wifi",
        name="Disable WiFi",
        description="Go to Settings and turn off WiFi if it's on.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="wifi_enabled",
        expected_result=False,
        tags=["settings", "wifi", "toggle", "off"],
    ),

    # Task 10: Disable Bluetooth
    CUATask(
        id="train_10_disable_bluetooth",
        name="Disable Bluetooth",
        description="Go to Settings and turn off Bluetooth if it's on.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="bluetooth_enabled",
        expected_result=False,
        tags=["settings", "bluetooth", "toggle", "off"],
    ),

    # Task 11: Set screen brightness to minimum
    CUATask(
        id="train_11_min_brightness",
        name="Minimum Brightness",
        description="Open Settings and set the screen brightness to the minimum level.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="brightness_level",
        expected_result=1,
        tags=["settings", "display", "brightness", "low"],
    ),

    # Task 12: Open WiFi settings page
    CUATask(
        id="train_12_open_wifi_settings",
        name="Open WiFi Settings",
        description="Open Settings and navigate to the WiFi settings page.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="current_activity",
        expected_result="wifi_settings",
        tags=["settings", "wifi", "navigation"],
    ),

    # Task 13: Open Bluetooth settings page
    CUATask(
        id="train_13_open_bluetooth_settings",
        name="Open Bluetooth Settings",
        description="Open Settings and navigate to the Bluetooth settings page.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="current_activity",
        expected_result="bluetooth_settings",
        tags=["settings", "bluetooth", "navigation"],
    ),

    # Task 14: Open Display settings page
    CUATask(
        id="train_14_open_display_settings",
        name="Open Display Settings",
        description="Open Settings and navigate to the Display settings page.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="current_activity",
        expected_result="display_settings",
        tags=["settings", "display", "navigation"],
    ),

    # Task 15: Open Sound settings page
    CUATask(
        id="train_15_open_sound_settings",
        name="Open Sound Settings",
        description="Open Settings and navigate to the Sound settings page.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="current_activity",
        expected_result="sound_settings",
        tags=["settings", "sound", "navigation"],
    ),

    # Task 16: Increase media volume to maximum
    CUATask(
        id="train_16_max_media_volume",
        name="Max Media Volume",
        description="Open Settings or use volume controls to set the media volume to maximum.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=100,
        tags=["settings", "sound", "volume", "media"],
    ),

    # Task 17: Mute media volume
    CUATask(
        id="train_17_mute_media_volume",
        name="Mute Media Volume",
        description="Open Settings or use volume controls to mute the media volume (set to 0).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=0,
        tags=["settings", "sound", "volume", "media", "mute"],
    ),

    # Task 18: Enable auto-rotate
    CUATask(
        id="train_18_enable_auto_rotate",
        name="Enable Auto-Rotate",
        description="Go to Display settings or quick settings and enable screen auto-rotate.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="auto_rotate_enabled",
        expected_result=True,
        tags=["settings", "display", "rotation"],
    ),

    # Task 19: Disable auto-rotate
    CUATask(
        id="train_19_disable_auto_rotate",
        name="Disable Auto-Rotate",
        description="Go to Display settings or quick settings and disable screen auto-rotate.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="auto_rotate_enabled",
        expected_result=False,
        tags=["settings", "display", "rotation", "off"],
    ),

    # Task 20: Open recent apps overview
    CUATask(
        id="train_20_open_recents",
        name="Open Recent Apps",
        description="Open the recent apps overview screen.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.NAVIGATION,
        max_steps=4,
        validation_type="state",
        validation_query="is_recents_screen",
        expected_result=True,
        tags=["navigation", "recents"],
    ),

    # Task 21: Switch from one app back to Settings
    CUATask(
        id="train_21_switch_to_settings",
        name="Switch to Settings from Recents",
        description="From the recent apps overview, select and open the Settings app.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.NAVIGATION,
        max_steps=7,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.settings",
        tags=["navigation", "recents", "multitask"],
    ),

    # Task 22: Open Phone dialer
    CUATask(
        id="train_22_open_dialer",
        name="Open Phone Dialer",
        description="From the home screen, open the Phone dialer app.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.dialer",
        tags=["app", "phone", "dialer"],
    ),

    # Task 23: Open Messages app
    CUATask(
        id="train_23_open_messages",
        name="Open Messages",
        description="From the home screen, open the default Messages app.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.messaging",
        tags=["app", "messages", "sms"],
    ),

    # Task 24: Open Camera app
    CUATask(
        id="train_24_open_camera",
        name="Open Camera",
        description="From the home screen, open the Camera app.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.camera",
        tags=["app", "camera"],
    ),

    # Task 25: Open Gallery/Photos app
    CUATask(
        id="train_25_open_gallery",
        name="Open Gallery",
        description="From the home screen, open the Gallery or Photos app.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=6,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.gallery",
        tags=["app", "gallery", "photos"],
    ),

    # Task 26: Open app drawer and find Settings icon
    CUATask(
        id="train_26_find_settings_in_drawer",
        name="Find Settings in App Drawer",
        description="Open the app drawer and scroll if needed to locate the Settings app icon.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.NAVIGATION,
        max_steps=8,
        validation_type="state",
        validation_query="found_icon",
        expected_result="com.android.settings",
        tags=["navigation", "launcher", "search"],
    ),

    # Task 27: Search app by name in launcher and open it
    CUATask(
        id="train_27_search_app_in_launcher",
        name="Search App in Launcher",
        description="Use the launcher search to search for the Chrome app by name and open it.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.NAVIGATION,
        max_steps=9,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.chrome",
        tags=["navigation", "launcher", "search", "chrome"],
    ),

    # Task 28: Open quick settings panel
    CUATask(
        id="train_28_open_quick_settings",
        name="Open Quick Settings",
        description="Swipe down from the top of the screen to open the quick settings panel.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.NAVIGATION,
        max_steps=4,
        validation_type="state",
        validation_query="is_quick_settings_open",
        expected_result=True,
        tags=["navigation", "system", "quick_settings"],
    ),

    # Task 29: Toggle WiFi from quick settings
    CUATask(
        id="train_29_toggle_wifi_quick_settings",
        name="Toggle WiFi from Quick Settings",
        description="Use the quick settings panel to toggle the WiFi state (turn it on if off).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=7,
        validation_type="state",
        validation_query="wifi_enabled",
        expected_result=True,
        tags=["settings", "wifi", "quick_settings"],
    ),

    # Task 30: Toggle Bluetooth from quick settings
    CUATask(
        id="train_30_toggle_bluetooth_quick_settings",
        name="Toggle Bluetooth from Quick Settings",
        description="Use the quick settings panel to toggle the Bluetooth state (turn it on if off).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=7,
        validation_type="state",
        validation_query="bluetooth_enabled",
        expected_result=True,
        tags=["settings", "bluetooth", "quick_settings"],
    ),

    # Task 31: Enable Do Not Disturb from quick settings
    CUATask(
        id="train_31_enable_dnd_quick_settings",
        name="Enable Do Not Disturb from Quick Settings",
        description="Use the quick settings panel to enable Do Not Disturb mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=7,
        validation_type="state",
        validation_query="dnd_enabled",
        expected_result=True,
        tags=["settings", "notifications", "dnd", "quick_settings"],
    ),

    # Task 32: Disable Do Not Disturb from quick settings
    CUATask(
        id="train_32_disable_dnd_quick_settings",
        name="Disable Do Not Disturb from Quick Settings",
        description="Use the quick settings panel to disable Do Not Disturb mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=7,
        validation_type="state",
        validation_query="dnd_enabled",
        expected_result=False,
        tags=["settings", "notifications", "dnd", "quick_settings", "off"],
    ),

    # Task 33: Open Notifications shade only
    CUATask(
        id="train_33_open_notifications",
        name="Open Notifications",
        description="Swipe down from the top of the screen to open the notifications shade (not full quick settings).",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.NAVIGATION,
        max_steps=4,
        validation_type="state",
        validation_query="is_notifications_open",
        expected_result=True,
        tags=["navigation", "notifications"],
    ),

    # Task 34: Clear all notifications
    CUATask(
        id="train_34_clear_notifications",
        name="Clear All Notifications",
        description="Open the notifications shade and clear all existing notifications.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=8,
        validation_type="state",
        validation_query="notification_count",
        expected_result=0,
        tags=["system", "notifications", "clear"],
    ),

    # Task 35: Change screen timeout to 30 seconds
    CUATask(
        id="train_35_timeout_30s",
        name="Change Screen Timeout to 30s",
        description="Open Settings, navigate to Display, and set the screen timeout to 30 seconds.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=30000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 36: Change screen timeout to 1 minute
    CUATask(
        id="train_36_timeout_1min",
        name="Change Screen Timeout to 1 Minute",
        description="Open Settings, navigate to Display, and set the screen timeout to 1 minute.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=60000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 37: Change screen timeout to 2 minutes
    CUATask(
        id="train_37_timeout_2min",
        name="Change Screen Timeout to 2 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 2 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=120000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 38: Open Language & Input settings
    CUATask(
        id="train_38_open_language_input",
        name="Open Language & Input Settings",
        description="Open Settings and navigate to the Language & Input (or System > Languages & input) page.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="current_activity",
        expected_result="language_input_settings",
        tags=["settings", "language", "input"],
    ),

    # Task 39: Open Date & Time settings
    CUATask(
        id="train_39_open_date_time",
        name="Open Date & Time Settings",
        description="Open Settings and navigate to the Date & Time settings page.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="current_activity",
        expected_result="date_time_settings",
        tags=["settings", "system", "time"],
    ),

    # Task 40: Enable automatic date & time
    CUATask(
        id="train_40_enable_auto_time",
        name="Enable Automatic Date & Time",
        description="In Date & Time settings, enable automatic date & time (network-provided time).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="auto_time_enabled",
        expected_result=True,
        tags=["settings", "system", "time", "auto"],
    ),

    # Task 41: Disable automatic date & time
    CUATask(
        id="train_41_disable_auto_time",
        name="Disable Automatic Date & Time",
        description="In Date & Time settings, disable automatic date & time.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="auto_time_enabled",
        expected_result=False,
        tags=["settings", "system", "time", "manual"],
    ),

    # Task 42: Enable battery saver
    CUATask(
        id="train_42_enable_battery_saver",
        name="Enable Battery Saver",
        description="Open Settings, go to Battery, and enable Battery Saver mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="battery_saver_enabled",
        expected_result=True,
        tags=["settings", "battery", "saver"],
    ),

    # Task 43: Disable battery saver
    CUATask(
        id="train_43_disable_battery_saver",
        name="Disable Battery Saver",
        description="Open Settings, go to Battery, and disable Battery Saver mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="battery_saver_enabled",
        expected_result=False,
        tags=["settings", "battery", "saver", "off"],
    ),

    # Task 44: Check storage usage
    CUATask(
        id="train_44_open_storage_settings",
        name="Open Storage Settings",
        description="Open Settings and navigate to the Storage settings page to view device storage usage.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="current_activity",
        expected_result="storage_settings",
        tags=["settings", "storage", "info"],
    ),

    # Task 45: Open App Info for Chrome
    CUATask(
        id="train_45_open_chrome_app_info",
        name="Open Chrome App Info",
        description="From Settings > Apps (or long-press), open the App Info page for Chrome.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.APP,
        max_steps=14,
        validation_type="state",
        validation_query="current_activity",
        expected_result="app_info_com.android.chrome",
        tags=["settings", "apps", "chrome", "info"],
    ),

    # Task 46: Force stop Chrome from App Info
    CUATask(
        id="train_46_force_stop_chrome",
        name="Force Stop Chrome",
        description="Navigate to Chrome's App Info page and tap Force Stop to stop the app.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_running_com.android.chrome",
        expected_result=False,
        tags=["settings", "apps", "chrome", "force_stop"],
    ),

    # Task 47: Clear Chrome app cache only
    CUATask(
        id="train_47_clear_chrome_cache",
        name="Clear Chrome App Cache",
        description="Navigate to Chrome's App Info > Storage and clear only the app cache (not data).",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=18,
        validation_type="state",
        validation_query="chrome_cache_cleared",
        expected_result=True,
        tags=["settings", "apps", "chrome", "cache"],
    ),

    # Task 48: Go to primary home page
    CUATask(
        id="train_48_go_to_primary_home_page",
        name="Go to Primary Home Page",
        description="From any secondary home screen page, navigate back to the primary (center) home screen.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.NAVIGATION,
        max_steps=6,
        validation_type="state",
        validation_query="is_primary_home",
        expected_result=True,
        tags=["navigation", "launcher", "home"],
    ),

    # Task 49: Open a website in Chrome
    CUATask(
        id="train_49_open_website_chrome",
        name="Open Website in Chrome",
        description="Open Chrome, tap the address bar, input a URL (e.g., example.com), and navigate to it.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=18,
        validation_type="state",
        validation_query="current_url_domain",
        expected_result="example.com",
        tags=["browser", "chrome", "input", "url"],
    ),

    # Task 50: Search in Chrome using search bar
    CUATask(
        id="train_50_search_in_chrome",
        name="Search in Chrome",
        description="Open Chrome and perform a web search for the text 'Android test'.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=18,
        validation_type="state",
        validation_query="search_query",
        expected_result="Android test",
        tags=["browser", "chrome", "input", "search"],
    ),

    # Task 51: Compose a new SMS (no need to send)
    CUATask(
        id="train_51_compose_sms",
        name="Compose SMS",
        description="Open the Messages app, start composing a new message, and type some text in the message field.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=18,
        validation_type="state",
        validation_query="is_composing_message",
        expected_result=True,
        tags=["messages", "sms", "input"],
    ),

    # Task 52: Add a contact from Phone app
    CUATask(
        id="train_52_add_contact_from_phone",
        name="Add Contact from Phone App",
        description="Open the Phone app, go to Contacts or add a new contact with a name and phone number (dummy data).",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=20,
        validation_type="state",
        validation_query="last_created_contact",
        expected_result="test_contact",
        tags=["contacts", "phone", "input"],
    ),

    # Task 53: Open Contacts app
    CUATask(
        id="train_53_open_contacts",
        name="Open Contacts",
        description="From the home screen, open the Contacts app.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=6,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.contacts",
        tags=["app", "contacts"],
    ),

    # Task 54: Search a contact in Contacts app
    CUATask(
        id="train_54_search_contact",
        name="Search Contact",
        description="Open Contacts and use the search field to search for a contact name (e.g., 'test').",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=18,
        validation_type="state",
        validation_query="last_contact_search_query",
        expected_result="test",
        tags=["contacts", "search", "input"],
    ),

    # Task 55: Open Screen Lock settings
    CUATask(
        id="train_55_open_screen_lock_settings",
        name="Open Screen Lock Settings",
        description="Open Settings, go to Security (or Security & privacy), and open the Screen lock settings page.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=18,
        validation_type="state",
        validation_query="current_activity",
        expected_result="screen_lock_settings",
        tags=["settings", "security", "lock"],
    ),

    # Task 56: Enable location/GPS
    CUATask(
        id="train_56_enable_location",
        name="Enable Location",
        description="Open Settings or quick settings and enable Location (GPS).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="location_enabled",
        expected_result=True,
        tags=["settings", "location", "gps"],
    ),

    # Task 57: Disable location/GPS
    CUATask(
        id="train_57_disable_location",
        name="Disable Location",
        description="Open Settings or quick settings and disable Location (GPS).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="location_enabled",
        expected_result=False,
        tags=["settings", "location", "gps", "off"],
    ),

    # Task 58: Connect to a known WiFi network
    CUATask(
        id="train_58_connect_known_wifi",
        name="Connect to Known WiFi",
        description="Open WiFi settings and connect to an already-saved WiFi network from the list.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="wifi_connected",
        expected_result=True,
        tags=["settings", "wifi", "connect"],
    ),

    # Task 59: Forget a saved WiFi network
    CUATask(
        id="train_59_forget_wifi_network",
        name="Forget WiFi Network",
        description="Open WiFi settings, open a saved network details, and forget/remove the network.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="wifi_network_forgotten",
        expected_result=True,
        tags=["settings", "wifi", "forget"],
    ),

    # Task 60: Change system language
    CUATask(
        id="train_60_change_system_language",
        name="Change System Language",
        description="Open Language & input settings and change the system language to another language (e.g., English if not already).",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="system_language",
        expected_result="en",
        tags=["settings", "language", "system"],
    ),

    # Task 61: Enable dark theme
    CUATask(
        id="train_61_enable_dark_theme",
        name="Enable Dark Theme",
        description="Open Display settings or quick settings and enable the system dark theme.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="dark_theme_enabled",
        expected_result=True,
        tags=["settings", "display", "theme", "dark"],
    ),

    # Task 62: Disable dark theme
    CUATask(
        id="train_62_disable_dark_theme",
        name="Disable Dark Theme",
        description="Open Display settings or quick settings and disable the system dark theme (switch to light).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="dark_theme_enabled",
        expected_result=False,
        tags=["settings", "display", "theme", "light"],
    ),

    # Task 63: Take a screenshot
    CUATask(
        id="train_63_take_screenshot",
        name="Take Screenshot",
        description="Trigger a screenshot capture using the device controls or UI, and confirm that a new screenshot exists.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=15,
        validation_type="state",
        validation_query="last_screenshot_taken",
        expected_result=True,
        tags=["system", "screenshot"],
    ),

    # Task 64: Open recent screenshot in Gallery
    CUATask(
        id="train_64_open_last_screenshot",
        name="Open Last Screenshot",
        description="Open the Gallery/Photos app and open the most recent screenshot image.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.NAVIGATION,
        max_steps=18,
        validation_type="state",
        validation_query="current_media_type",
        expected_result="image_screenshot",
        tags=["gallery", "photos", "screenshot"],
    ),

    # Task 65: Uninstall an app from home screen
    CUATask(
        id="train_65_uninstall_app_home",
        name="Uninstall App from Home Screen",
        description="From the home screen, long-press an app icon (e.g., a test app) and uninstall it.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=20,
        validation_type="state",
        validation_query="app_installed_test_app",
        expected_result=False,
        tags=["system", "apps", "uninstall"],
    ),

    # Task 66: Pin an app to the home screen
    CUATask(
        id="train_66_pin_app_to_home",
        name="Pin App to Home Screen",
        description="Open the app drawer, long-press an app icon, and place it on the home screen.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.NAVIGATION,
        max_steps=18,
        validation_type="state",
        validation_query="icon_on_home",
        expected_result="test_app",
        tags=["launcher", "apps", "home"],
    ),

    # Task 67: Move an app icon between home screen pages
    CUATask(
        id="train_67_move_icon_between_pages",
        name="Move Icon Between Home Pages",
        description="Drag an app icon from one home screen page to another page.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.NAVIGATION,
        max_steps=18,
        validation_type="state",
        validation_query="icon_page_index",
        expected_result=1,
        tags=["launcher", "apps", "home", "drag"],
    ),

    # Task 68: Create an app folder on home screen
    CUATask(
        id="train_68_create_app_folder",
        name="Create App Folder",
        description="On the home screen, drag one app icon onto another to create a folder containing both apps.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.NAVIGATION,
        max_steps=20,
        validation_type="state",
        validation_query="folder_created",
        expected_result=True,
        tags=["launcher", "folder", "home"],
    ),

    # Task 69: Rename an app folder on home screen
    CUATask(
        id="train_69_rename_folder",
        name="Rename App Folder",
        description="Open an existing app folder on the home screen and rename it to 'Tools'.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=18,
        validation_type="state",
        validation_query="folder_name",
        expected_result="Tools",
        tags=["launcher", "folder", "home", "rename"],
    ),

    # Task 70: Change wallpaper from Settings
    CUATask(
        id="train_70_change_wallpaper_settings",
        name="Change Wallpaper from Settings",
        description="Open Settings, go to Display or Wallpaper, and change the home screen wallpaper.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="wallpaper_changed",
        expected_result=True,
        tags=["settings", "display", "wallpaper"],
    ),

    # Task 71: Change wallpaper via long-press on home screen
    CUATask(
        id="train_71_change_wallpaper_home",
        name="Change Wallpaper from Home Screen",
        description="Long-press on the home screen, open wallpaper picker, and apply a new wallpaper.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.NAVIGATION,
        max_steps=22,
        validation_type="state",
        validation_query="wallpaper_changed",
        expected_result=True,
        tags=["launcher", "wallpaper", "home"],
    ),

    # Task 72: Enable airplane mode from quick settings
    CUATask(
        id="train_72_enable_airplane_quick",
        name="Enable Airplane Mode from Quick Settings",
        description="Use quick settings to enable airplane mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="airplane_mode",
        expected_result=True,
        tags=["settings", "network", "airplane", "quick_settings"],
    ),

    # Task 73: Disable airplane mode from quick settings
    CUATask(
        id="train_73_disable_airplane_quick",
        name="Disable Airplane Mode from Quick Settings",
        description="Use quick settings to disable airplane mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="airplane_mode",
        expected_result=False,
        tags=["settings", "network", "airplane", "quick_settings", "off"],
    ),

    # Task 74: Open Developer Options
    CUATask(
        id="train_74_open_developer_options",
        name="Open Developer Options",
        description="Open Settings, navigate to System > Developer options page (assuming it is already enabled).",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="current_activity",
        expected_result="developer_options_settings",
        tags=["settings", "system", "developer"],
    ),

    # Task 75: Enable USB debugging in Developer Options
    CUATask(
        id="train_75_enable_usb_debugging",
        name="Enable USB Debugging",
        description="In Developer options, enable USB debugging.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="usb_debugging_enabled",
        expected_result=True,
        tags=["settings", "developer", "usb_debugging"],
    ),

    # Task 76: Disable USB debugging in Developer Options
    CUATask(
        id="train_76_disable_usb_debugging",
        name="Disable USB Debugging",
        description="In Developer options, disable USB debugging.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="usb_debugging_enabled",
        expected_result=False,
        tags=["settings", "developer", "usb_debugging", "off"],
    ),

    # Task 77: Enable app notifications for a specific app
    CUATask(
        id="train_77_enable_app_notifications",
        name="Enable App Notifications",
        description="Open Settings > Apps > [Test App] > Notifications and enable notifications for this app.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="test_app_notifications_enabled",
        expected_result=True,
        tags=["settings", "apps", "notifications"],
    ),

    # Task 78: Disable app notifications for a specific app
    CUATask(
        id="train_78_disable_app_notifications",
        name="Disable App Notifications",
        description="Open Settings > Apps > [Test App] > Notifications and disable notifications for this app.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="test_app_notifications_enabled",
        expected_result=False,
        tags=["settings", "apps", "notifications", "off"],
    ),

    # Task 79: Enable vibration for incoming calls
    CUATask(
        id="train_79_enable_call_vibration",
        name="Enable Call Vibration",
        description="Open Sound settings and enable vibration for incoming calls.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=14,
        validation_type="state",
        validation_query="call_vibration_enabled",
        expected_result=True,
        tags=["settings", "sound", "vibration"],
    ),

    # Task 80: Disable vibration for incoming calls
    CUATask(
        id="train_80_disable_call_vibration",
        name="Disable Call Vibration",
        description="Open Sound settings and disable vibration for incoming calls.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=14,
        validation_type="state",
        validation_query="call_vibration_enabled",
        expected_result=False,
        tags=["settings", "sound", "vibration", "off"],
    ),
]

# =============================================================================
# EVALUATION TASKS (20 tasks)
# =============================================================================

EVAL_TASKS = [
    # Eval Task 1: Change display timeout to 5 minutes
    CUATask(
        id="eval_01_display_timeout_5min",
        name="Change Display Timeout to 5 Minutes",
        description="Open Settings, go to Display, and change the screen timeout to 5 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=300000,
        tags=["settings", "display", "timeout"],
    ),

    # Eval Task 2: Enable Do Not Disturb
    CUATask(
        id="eval_02_dnd_mode_enable",
        name="Enable Do Not Disturb",
        description="Enable Do Not Disturb mode from quick settings or the Settings app.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="dnd_enabled",
        expected_result=True,
        tags=["settings", "notifications", "dnd"],
    ),

    # Eval Task 3: Disable Do Not Disturb
    CUATask(
        id="eval_03_dnd_mode_disable",
        name="Disable Do Not Disturb",
        description="Disable Do Not Disturb mode from quick settings or the Settings app.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="dnd_enabled",
        expected_result=False,
        tags=["settings", "notifications", "dnd", "off"],
    ),

    # Eval Task 4: Change system language
    CUATask(
        id="eval_04_change_language",
        name="Change System Language",
        description="Open Language & input settings and change the system language to Spanish.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=24,
        validation_type="state",
        validation_query="system_language",
        expected_result="es",
        tags=["settings", "language", "system"],
    ),

    # Eval Task 5: Enable dark theme
    CUATask(
        id="eval_05_enable_dark_theme",
        name="Enable Dark Theme",
        description="Enable the system dark theme from Display settings or quick settings.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="dark_theme_enabled",
        expected_result=True,
        tags=["settings", "display", "theme", "dark"],
    ),

    # Eval Task 6: Disable dark theme
    CUATask(
        id="eval_06_disable_dark_theme",
        name="Disable Dark Theme",
        description="Disable the system dark theme and switch to light theme.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="dark_theme_enabled",
        expected_result=False,
        tags=["settings", "display", "theme", "light"],
    ),

    # Eval Task 7: Enable battery saver
    CUATask(
        id="eval_07_enable_battery_saver",
        name="Enable Battery Saver",
        description="Open Battery settings and enable Battery Saver mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=14,
        validation_type="state",
        validation_query="battery_saver_enabled",
        expected_result=True,
        tags=["settings", "battery", "saver"],
    ),

    # Eval Task 8: Disable battery saver
    CUATask(
        id="eval_08_disable_battery_saver",
        name="Disable Battery Saver",
        description="Open Battery settings and disable Battery Saver mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=14,
        validation_type="state",
        validation_query="battery_saver_enabled",
        expected_result=False,
        tags=["settings", "battery", "saver", "off"],
    ),

    # Eval Task 9: Connect to known WiFi network
    CUATask(
        id="eval_09_connect_known_wifi",
        name="Connect to Known WiFi Network",
        description="Open WiFi settings and connect to a known WiFi network from the list.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="wifi_connected",
        expected_result=True,
        tags=["settings", "wifi", "connect"],
    ),

    # Eval Task 10: Forget saved WiFi network
    CUATask(
        id="eval_10_forget_wifi",
        name="Forget WiFi Network",
        description="In WiFi settings, open a saved network and forget/remove it.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="wifi_network_forgotten",
        expected_result=True,
        tags=["settings", "wifi", "forget"],
    ),

    # Eval Task 11: Change wallpaper
    CUATask(
        id="eval_11_change_wallpaper",
        name="Change Wallpaper",
        description="Change the home screen wallpaper via Settings or by long-pressing the home screen.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="wallpaper_changed",
        expected_result=True,
        tags=["settings", "display", "wallpaper"],
    ),

    # Eval Task 12: Open Developer Options
    CUATask(
        id="eval_12_open_developer_options",
        name="Open Developer Options",
        description="Open the Developer options page from Settings.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="current_activity",
        expected_result="developer_options_settings",
        tags=["settings", "system", "developer"],
    ),

    # Eval Task 13: Enable USB debugging
    CUATask(
        id="eval_13_enable_usb_debugging",
        name="Enable USB Debugging",
        description="In Developer options, enable USB debugging.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="usb_debugging_enabled",
        expected_result=True,
        tags=["settings", "developer", "usb_debugging"],
    ),

    # Eval Task 14: Disable USB debugging
    CUATask(
        id="eval_14_disable_usb_debugging",
        name="Disable USB Debugging",
        description="In Developer options, disable USB debugging.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="usb_debugging_enabled",
        expected_result=False,
        tags=["settings", "developer", "usb_debugging", "off"],
    ),

    # Eval Task 15: Enable app notifications
    CUATask(
        id="eval_15_enable_app_notifications",
        name="Enable App Notifications",
        description="Open Settings > Apps > [Test App] > Notifications and enable notifications.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="test_app_notifications_enabled",
        expected_result=True,
        tags=["settings", "apps", "notifications"],
    ),

    # Eval Task 16: Disable app notifications
    CUATask(
        id="eval_16_disable_app_notifications",
        name="Disable App Notifications",
        description="Open Settings > Apps > [Test App] > Notifications and disable notifications.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SETTINGS,
        max_steps=20,
        validation_type="state",
        validation_query="test_app_notifications_enabled",
        expected_result=False,
        tags=["settings", "apps", "notifications", "off"],
    ),

    # Eval Task 17: Take and open a screenshot
    CUATask(
        id="eval_17_take_and_open_screenshot",
        name="Take and Open Screenshot",
        description="Capture a screenshot and then open it in the Gallery/Photos app.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=24,
        validation_type="state",
        validation_query="current_media_type",
        expected_result="image_screenshot",
        tags=["system", "screenshot", "gallery"],
    ),

    # Eval Task 18: Compose SMS and save as draft
    CUATask(
        id="eval_18_compose_sms_draft",
        name="Compose SMS Draft",
        description="Open Messages, compose a new SMS with some text, and leave it as a draft.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=22,
        validation_type="state",
        validation_query="has_message_draft",
        expected_result=True,
        tags=["messages", "sms", "draft"],
    ),

    # Eval Task 19: Change media volume
    CUATask(
        id="eval_19_change_media_volume",
        name="Change Media Volume",
        description="Open Sound settings or use volume controls to set media volume to a specific non-zero level (e.g., 50%).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=14,
        validation_type="state",
        validation_query="media_volume",
        expected_result=50,
        tags=["settings", "sound", "volume", "media"],
    ),

    # Eval Task 20: Enable auto-rotate
    CUATask(
        id="eval_20_enable_auto_rotate",
        name="Enable Auto-Rotate",
        description="Enable screen auto-rotate from Display or quick settings.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="auto_rotate_enabled",
        expected_result=True,
        tags=["settings", "display", "rotation"],
    ),
]

# =============================================================================
# TASK UTILITIES
# =============================================================================

def get_all_tasks() -> List[CUATask]:
    """Get all tasks (training + evaluation)."""
    return TRAINING_TASKS + EVAL_TASKS


def get_training_tasks() -> List[CUATask]:
    """Get training tasks only."""
    return TRAINING_TASKS


def get_eval_tasks() -> List[CUATask]:
    """Get evaluation tasks only."""
    return EVAL_TASKS


def get_task_by_id(task_id: str) -> Optional[CUATask]:
    """Get a task by its ID."""
    for task in get_all_tasks():
        if task.id == task_id:
            return task
    return None


def get_tasks_by_category(category: TaskCategory) -> List[CUATask]:
    """Get tasks by category."""
    return [t for t in get_all_tasks() if t.category == category]


def get_tasks_by_difficulty(difficulty: TaskDifficulty) -> List[CUATask]:
    """Get tasks by difficulty."""
    return [t for t in get_all_tasks() if t.difficulty == difficulty]


# =============================================================================
# TASK PROMPT GENERATION
# =============================================================================

def create_task_prompt(task: CUATask) -> str:
    """Create a prompt for the task."""
    return f"""Task: {task.name}

{task.description}

You have a maximum of {task.max_steps} steps to complete this task.
Report task_complete when done, indicating success or failure."""


def create_task_system_prompt(task: CUATask) -> str:
    """Create system prompt for the task."""
    return f"""You are a helpful AI assistant that controls an Android device to complete tasks.

Your goal: {task.name}
Description: {task.description}

You can perform the following actions:
- click: Click on a UI element
- swipe: Swipe from one point to another  
- scroll: Scroll in a direction
- input: Type text into a field
- button_press: Press a system button (back, home, menu)
- task_complete: Report when the task is done

Guidelines:
1. Analyze the screenshot to understand the current screen state
2. Plan your next action based on the goal
3. Be efficient - use the minimum number of steps
4. Report task_complete with success=true when done, or success=false if stuck

Maximum steps allowed: {task.max_steps}"""


# =============================================================================
# AReaL 格式导出
# =============================================================================

def create_areal_task_dataset(tasks: List[CUATask]) -> List[Dict]:
    """将 CUA 任务转换为 AReaL 数据集格式。
    
    注意：这里只存储任务描述，实际的提示语生成由 gbox-cua 的 create_system_prompt 处理。
    
    Args:
        tasks: CUA 任务列表
        
    Returns:
        AReaL 格式的任务列表
    """
    # 导入 gbox-cua 的提示语生成函数
    from gbox_cua.prompts import create_system_prompt
    
    dataset = []
    for task in tasks:
        # 使用 gbox-cua 的提示语生成逻辑
        prompt = create_system_prompt(
            task_description=task.description,
            max_turns=task.max_steps,
        )
        dataset.append({
            "id": task.id,
            "task": task,
            "prompt": prompt,
            "metadata": {
                "name": task.name,
                "difficulty": task.difficulty.value,
                "category": task.category.value,
                "max_steps": task.max_steps,
            },
        })
    return dataset


def get_areal_train_dataset():
    """获取 AReaL 格式的训练数据集。"""
    return create_areal_task_dataset(get_training_tasks())


def get_areal_eval_dataset():
    """获取 AReaL 格式的评估数据集。"""
    return create_areal_task_dataset(get_eval_tasks())


__all__ = [
    "CUATask",
    "TaskDifficulty", 
    "TaskCategory",
    "TRAINING_TASKS",
    "EVAL_TASKS",
    "get_all_tasks",
    "get_training_tasks",
    "get_eval_tasks",
    "get_task_by_id",
    "get_tasks_by_category",
    "get_tasks_by_difficulty",
    "create_task_prompt",
    "create_task_system_prompt",
    "create_areal_task_dataset",
    "get_areal_train_dataset",
    "get_areal_eval_dataset",
]

