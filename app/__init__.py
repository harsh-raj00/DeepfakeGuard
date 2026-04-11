"""
=============================================================================
 Flask Application Factory
 Creates and configures the Flask app with SocketIO support.
=============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from flask import Flask
from flask_socketio import SocketIO
from flask_login import LoginManager

# ── Global SocketIO instance ───────────────────────────────────────────────
socketio = SocketIO()
login_manager = LoginManager()


def create_app():
    """
    Application factory pattern.
    Creates and fully configures the Flask application.

    Returns:
        Flask: Configured Flask application.
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static')
    )

    # ── Configuration ──
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG

    # ── Initialize extensions ──
    socketio.init_app(app, cors_allowed_origins="*", async_mode='threading')
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'

    # ── User loader for Flask-Login ──
    @login_manager.user_loader
    def load_user(user_id):
        from utils.db_utils import get_db_connection
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        if user:
            return User(dict(user))
        return None

    # ── Register blueprints ──
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # ── Register SocketIO events ──
    from app import socketio_events  # noqa: F401

    return app


class User:
    """Minimal user class for Flask-Login integration."""

    def __init__(self, user_dict):
        self.id = user_dict['id']
        self.username = user_dict['username']
        self.email = user_dict['email']
        self.is_active_user = user_dict.get('is_active', True)

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return self.is_active_user

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)
