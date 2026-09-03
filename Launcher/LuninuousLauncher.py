import os
import json
import math
import time
from datetime import datetime

from kivy.utils import platform
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ListProperty
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.chip import MDChip
from kivymd.uix.list import TwoLineAvatarIconListItem, IconLeftWidget, IconRightWidget
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.boxlayout import MDBoxLayout
 
class AggressiveContextualNN:
    """
    On-Device 2-Layer Neural Network with Online Backpropagation.
    Inputs (5-dim vector):
      1. Normalized Hour of Day (0.0 to 1.0)
      2. Normalized Day of Week (0.0 to 1.0)
      3. Normalized Last App Index (0.0 to 1.0)
      4. Time Delta ratio since last launch
      5. Global Launch Frequency ratio
    Outputs:
      Softmax probability distribution across all installed apps.
    """
    def __init__(self, num_apps, hidden_size=24, learning_rate=0.35):
        self.num_apps = max(num_apps, 1)
        self.hidden_size = hidden_size
        self.lr = learning_rate  # Aggressive learning rate
        self.input_size = 5
        
        # Initialize Weights & Biases
        self.W1 = [[(hash((i, j)) % 100) / 500.0 - 0.1 for j in range(hidden_size)] for i in range(self.input_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[(hash((i, j)) % 100) / 500.0 - 0.1 for j in range(self.num_apps)] for i in range(hidden_size)]
        self.b2 = [0.0] * self.num_apps

    def softmax(self, vec):
        max_val = max(vec) if vec else 0.0
        exps = [math.exp(v - max_val) for v in vec]
        sum_exps = sum(exps) or 1.0
        return [e / sum_exps for e in exps]

    def forward(self, x):
        self.x = x
        # Layer 1 (ReLU activation)
        self.z1 = [sum(x[i] * self.W1[i][j] for i in range(self.input_size)) + self.b1[j] for j in range(self.hidden_size)]
        self.a1 = [max(0.0, val) for val in self.z1]
        
        # Layer 2 (Softmax activation)
        self.z2 = [sum(self.a1[i] * self.W2[i][j] for i in range(self.hidden_size)) + self.b2[j] for j in range(self.num_apps)]
        self.probs = self.softmax(self.z2)
        return self.probs

    def train_step(self, x, target_app_index):
        """Performs single-sample immediate online backpropagation gradient descent."""
        if target_app_index >= self.num_apps:
            return

        self.forward(x)
        dz2 = list(self.probs)
        dz2[target_app_index] -= 1.0
        dW2 = [[self.a1[i] * dz2[j] for j in range(self.num_apps)] for i in range(self.hidden_size)]
        db2 = dz2
        da1 = [sum(dz2[j] * self.W2[i][j] for j in range(self.num_apps)) for i in range(self.hidden_size)]
        dz1 = [da1[i] if self.z1[i] > 0 else 0.0 for i in range(self.hidden_size)]
        
        dW1 = [[x[i] * dz1[j] for j in range(self.hidden_size)] for i in range(self.input_size)]
        db1 = dz1
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.W1[i][j] -= self.lr * dW1[i][j]
        for i in range(self.hidden_size):
            self.b1[i] -= self.lr * db1[i]
            for j in range(self.num_apps):
                self.W2[i][j] -= self.lr * dW2[i][j]
        for j in range(self.num_apps):
            self.b2[j] -= self.lr * db2[j]

    def serialize(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def deserialize(self, data):
        if data:
            self.W1 = data.get("W1", self.W1)
            self.b1 = data.get("b1", self.b1)
            self.W2 = data.get("W2", self.W2)
            self.b2 = data.get("b2", self.b2)

KV_LAYOUT = """
MDScreen:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "ai.luninuous.launcher"
            elevation: 3
            right_action_items: [["theme-light-dark", lambda x: app.toggle_theme_style()]]

        MDBottomNavigation:
            id: nav_bar
            MDBottomNavigationItem:
                name: "screen_home"
                text: "Drawer"
                icon: "view-grid"

                MDBoxLayout:
                    orientation: "vertical"
                    padding: "12dp"
                    spacing: "10dp"
                    MDTextField:
                        id: search_field
                        hint_text: "Search apps..."
                        icon_left: "magnify"
                        mode: "rectangle"
                        size_hint_y: None
                        height: "48dp"
                        on_text: app.filter_apps(self.text)
                    MDLabel:
                        text: "⚡ AI Predicted Next Apps"
                        font_style: "Subtitle2"
                        adaptive_height: True

                    MDScrollView:
                        size_hint_y: None
                        height: "56dp"
                        do_scroll_y: False
                        MDBoxLayout:
                            id: ai_chip_container
                            orientation: "horizontal"
                            spacing: "8dp"
                            adaptive_width: True

                    MDSeparator:
                    MDScrollView:
                        id: drawer_scroll
                        MDBoxLayout:
                            id: main_drawer_container
                            orientation: "vertical"
                            adaptive_height: True
                            spacing: "10dp"
            MDBottomNavigationItem:
                name: "screen_studio"
                text: "Studio"
                icon: "palette"

                MDScrollView:
                    MDBoxLayout:
                        orientation: "vertical"
                        padding: "16dp"
                        spacing: "16dp"
                        adaptive_height: True

                        MDLabel:
                            text: "🎨 Hyper Theme Customization"
                            font_style: "H6"
                            adaptive_height: True

                        MDLabel:
                            text: "Primary Color Palette"
                            font_style: "Subtitle2"
                            adaptive_height: True

                        MDGridLayout:
                            cols: 3
                            spacing: "8dp"
                            adaptive_height: True

                            MDRaisedButton:
                                text: "Cyber Purple"
                                md_bg_color: 0.5, 0.1, 0.9, 1
                                on_release: app.set_color_palette("DeepPurple")

                            MDRaisedButton:
                                text: "Neon Cyan"
                                md_bg_color: 0.0, 0.7, 0.9, 1
                                on_release: app.set_color_palette("Cyan")

                            MDRaisedButton:
                                text: "Crimson Red"
                                md_bg_color: 0.9, 0.1, 0.3, 1
                                on_release: app.set_color_palette("Red")

                            MDRaisedButton:
                                text: "Emerald"
                                md_bg_color: 0.1, 0.7, 0.3, 1
                                on_release: app.set_color_palette("Green")

                            MDRaisedButton:
                                text: "Amber Gold"
                                md_bg_color: 0.9, 0.6, 0.0, 1
                                on_release: app.set_color_palette("Amber")

                            MDRaisedButton:
                                text: "Teal Matrix"
                                md_bg_color: 0.0, 0.5, 0.5, 1
                                on_release: app.set_color_palette("Teal")

                        MDSeparator:

                        MDLabel:
                            text: "Layout & Scale Engine"
                            font_style: "H6"
                            adaptive_height: True

                        MDLabel:
                            text: f"Grid Columns: {int(grid_slider.value)}"
                            adaptive_height: True

                        MDSlider:
                            id: grid_slider
                            min: 3
                            max: 6
                            value: app.grid_cols
                            step: 1
                            on_value: app.update_grid_cols(int(self.value))

                        MDLabel:
                            text: f"Icon Size: {int(icon_slider.value)}dp"
                            adaptive_height: True

                        MDSlider:
                            id: icon_slider
                            min: 36
                            max: 72
                            value: app.icon_size
                            step: 4
                            on_value: app.update_icon_size(int(self.value))

                        MDSeparator:

                        MDLabel:
                            text: "🧠 AI Aggressiveness Tuning"
                            font_style: "H6"
                            adaptive_height: True

                        MDLabel:
                            text: f"Learning Rate (Eta): {ai_slider.value:.2f}"
                            adaptive_height: True

                        MDSlider:
                            id: ai_slider
                            min: 0.05
                            max: 0.80
                            value: app.ai_learning_rate
                            step: 0.05
                            on_value: app.update_ai_learning_rate(self.value)

            MDBottomNavigationItem:
                name: "screen_security"
                text: "Privacy"
                icon: "shield-lock"

                MDScrollView:
                    MDBoxLayout:
                        orientation: "vertical"
                        padding: "16dp"
                        spacing: "12dp"
                        adaptive_height: True

                        MDLabel:
                            text: "🔒 App Hiding & Security"
                            font_style: "H6"
                            adaptive_height: True

                        MDLabel:
                            text: "Toggle visibility to hide applications from the main drawer."
                            font_style: "Caption"
                            adaptive_height: True

                        MDList:
                            id: privacy_app_list
"""


class LuninuousLauncherApp(MDApp):
    grid_cols = NumericProperty(4)
    icon_size = NumericProperty(48)
    ai_learning_rate = NumericProperty(0.35)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.installed_apps = []
        self.hidden_apps = set()
        self.last_launched_index = 0
        self.last_launch_time = time.time()
        self.launch_counts = {}
        
        self.config_file = "luninuous_config.json"
        self.weights_file = "ai_nn_weights.json"

    def build(self):
        self.title = "ai.luninuous.launcher"
        self.load_user_config()
        self.request_android_permissions()
        return Builder.load_string(KV_LAYOUT)

    def on_start(self):
        self.installed_apps = self.get_installed_apps()
        self.nn = AggressiveContextualNN(
            num_apps=len(self.installed_apps), 
            learning_rate=self.ai_learning_rate
        )
        self.load_ai_weights()
        self.refresh_all_views()

    def request_android_permissions(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE,
            ])

    def load_user_config(self):
        path = os.path.join(self.user_data_dir, self.config_file)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    cfg = json.load(f)
                    self.theme_cls.primary_palette = cfg.get("primary_palette", "DeepPurple")
                    self.theme_cls.theme_style = cfg.get("theme_style", "Dark")
                    self.grid_cols = cfg.get("grid_cols", 4)
                    self.icon_size = cfg.get("icon_size", 48)
                    self.ai_learning_rate = cfg.get("ai_learning_rate", 0.35)
                    self.hidden_apps = set(cfg.get("hidden_apps", []))
                    self.launch_counts = cfg.get("launch_counts", {})
            except Exception:
                pass
        else:
            self.theme_cls.primary_palette = "DeepPurple"
            self.theme_cls.theme_style = "Dark"

    def save_user_config(self):
        path = os.path.join(self.user_data_dir, self.config_file)
        with open(path, 'w') as f:
            json.dump({
                "primary_palette": self.theme_cls.primary_palette,
                "theme_style": self.theme_cls.theme_style,
                "grid_cols": self.grid_cols,
                "icon_size": self.icon_size,
                "ai_learning_rate": self.ai_learning_rate,
                "hidden_apps": list(self.hidden_apps),
                "launch_counts": self.launch_counts
            }, f)

    def save_ai_weights(self):
        path = os.path.join(self.user_data_dir, self.weights_file)
        with open(path, 'w') as f:
            json.dump(self.nn.serialize(), f)

    def load_ai_weights(self):
        path = os.path.join(self.user_data_dir, self.weights_file)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.nn.deserialize(json.load(f))
            except Exception:
                pass
    def set_color_palette(self, palette_name):
        self.theme_cls.primary_palette = palette_name
        self.save_user_config()

    def toggle_theme_style(self):
        self.theme_cls.theme_style = "Light" if self.theme_cls.theme_style == "Dark" else "Dark"
        self.save_user_config()

    def update_grid_cols(self, cols):
        self.grid_cols = cols
        self.save_user_config()
        self.refresh_all_views()

    def update_icon_size(self, size):
        self.icon_size = size
        self.save_user_config()
        self.refresh_all_views()

    def update_ai_learning_rate(self, lr):
        self.ai_learning_rate = lr
        if hasattr(self, 'nn'):
            self.nn.lr = lr
        self.save_user_config()

    def toggle_app_visibility(self, package_name):
        if package_name in self.hidden_apps:
            self.hidden_apps.remove(package_name)
        else:
            self.hidden_apps.add(package_name)
        self.save_user_config()
        self.refresh_all_views()

    # --- CONTEXT VECTOR & APP MANAGEMENT ---
    def get_context_vector(self):
        now = datetime.now()
        hour_norm = now.hour / 24.0
        day_norm = now.weekday() / 7.0
        last_app_norm = self.last_launched_index / max(len(self.installed_apps), 1)
        
        delta = time.time() - self.last_launch_time
        delta_norm = min(delta / 3600.0, 1.0)
        
        total_launches = sum(self.launch_counts.values()) or 1
        freq_norm = (self.launch_counts.get(str(self.last_launched_index), 0)) / total_launches
        
        return [hour_norm, day_norm, last_app_norm, delta_norm, freq_norm]

    def get_installed_apps(self):
        if platform == 'android':
            from jnius import autoclass
            PackageManager = autoclass('android.content.pm.PackageManager')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Context = PythonActivity.mActivity
            pm = Context.getPackageManager()
            
            Intent = autoclass('android.content.Intent')
            intent = Intent(Intent.ACTION_MAIN, None)
            intent.addCategory(Intent.CATEGORY_LAUNCHER)
            
            query = pm.queryIntentActivities(intent, 0)
            apps = []
            for i in range(query.size()):
                ri = query.get(i)
                apps.append({
                    "name": str(ri.loadLabel(pm)),
                    "package": str(ri.activityInfo.packageName),
                    "icon": "android"
                })
            return sorted(apps, key=lambda x: x['name'])
        return [
            {"name": "Camera", "package": "com.android.camera", "icon": "camera"},
            {"name": "Chrome", "package": "com.android.chrome", "icon": "web"},
            {"name": "Gallery", "package": "com.android.gallery", "icon": "image"},
            {"name": "Messages", "package": "com.android.mms", "icon": "message"},
            {"name": "Settings", "package": "com.android.settings", "icon": "cog"},
            {"name": "YouTube", "package": "com.google.youtube", "icon": "youtube"},
            {"name": "Music", "package": "com.android.music", "icon": "music"},
            {"name": "Calculator", "package": "com.android.calculator2", "icon": "calculator"},
            {"name": "Files", "package": "com.android.documentsui", "icon": "folder"},
            {"name": "Clock", "package": "com.android.deskclock", "icon": "clock"},
            {"name": "Contacts", "package": "com.android.contacts", "icon": "account-box"},
            {"name": "Maps", "package": "com.google.android.apps.maps", "icon": "map-marker"},
        ]
    def refresh_all_views(self, filter_text=""):
        ctx = self.get_context_vector()
        probs = self.nn.forward(ctx)
        
        # Rank apps by NN score
        ranked_apps = []
        for idx, app in enumerate(self.installed_apps):
            if app['package'] not in self.hidden_apps:
                if not filter_text or filter_text.lower() in app['name'].lower():
                    prob = probs[idx] if idx < len(probs) else 0.0
                    ranked_apps.append((app, idx, prob))
        
        ranked_apps.sort(key=lambda x: x[2], reverse=True)
        chip_box = self.root.ids.ai_chip_container
        chip_box.clear_widgets()
        for app, idx, score in ranked_apps[:4]:
            chip = MDChip(
                text=f"{app['name']} ({int(score*100)}%)",
                icon_left=app['icon'],
                on_release=lambda x, p=app['package'], i=idx: self.launch_app(p, i)
            )
            chip_box.add_widget(chip)
        drawer_container = self.root.ids.main_drawer_container
        drawer_container.clear_widgets()
        
        grid = MDGridLayout(
            cols=self.grid_cols,
            spacing="12dp",
            adaptive_height=True
        )
        
        for app, idx, score in ranked_apps:
            card = MDCard(
                orientation="vertical",
                size_hint_y=None,
                height=f"{self.icon_size + 40}dp",
                padding="4dp",
                ripple_behavior=True,
                on_release=lambda x, p=app['package'], i=idx: self.launch_app(p, i)
            )
            
            icon_btn = MDIconButton(
                icon=app['icon'],
                user_font_size=f"{self.icon_size}sp",
                pos_hint={"center_x": 0.5}
            )
            
            lbl = MDRaisedButton(
                text=app['name'],
                size_hint_x=1,
                elevation=0,
                font_size="10sp",
                md_bg_color=(0, 0, 0, 0)
            )
            
            card.add_widget(icon_btn)
            card.add_widget(lbl)
            grid.add_widget(card)
        drawer_container.add_widget(grid)
        privacy_list = self.root.ids.privacy_app_list
        privacy_list.clear_widgets()
        for app in self.installed_apps:
            is_hidden = app['package'] in self.hidden_apps
            item = TwoLineAvatarIconListItem(
                text=app['name'],
                secondary_text="Hidden from Drawer" if is_hidden else "Visible in Drawer",
                on_release=lambda x, p=app['package']: self.toggle_app_visibility(p)
            )
            item.add_widget(IconLeftWidget(icon=app['icon']))
            item.add_widget(IconRightWidget(icon="eye-off" if is_hidden else "eye"))
            privacy_list.add_widget(item)

    def filter_apps(self, text):
        self.refresh_all_views(filter_text=text)

    def launch_app(self, package_name, app_index):
        ctx = self.get_context_vector()
        self.nn.train_step(ctx, target_app_index=app_index)
        self.last_launched_index = app_index
        self.last_launch_time = time.time()
        self.launch_counts[str(app_index)] = self.launch_counts.get(str(app_index), 0) + 1
        self.save_ai_weights()
        self.save_user_config()
        self.refresh_all_views()
        if platform == 'android':
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Context = PythonActivity.mActivity
            pm = Context.getPackageManager()
            intent = pm.getLaunchIntentForPackage(package_name)
            if intent:
                Context.startActivity(intent)
if __name__ == "__main__":
    LuninuousLauncherApp().run()