"""Flask web application for SirenLedger dashboard and API."""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask.cli import with_appcontext
from flask_socketio import SocketIO

from ..config import ConfigManager
from ..storage import Database
from .api import create_api_blueprint
from .dashboard import create_dashboard_blueprint
from .websocket import WebSocketManager

logger = logging.getLogger(__name__)


def create_app(config_path: Optional[Path] = None) -> tuple[Flask, SocketIO]:
    """Create and configure Flask application with WebSocket support.
    
    Parameters
    ----------
    config_path : Optional[Path]
        Path to configuration file.
        
    Returns
    -------
    tuple[Flask, SocketIO]
        Configured Flask application and SocketIO instance.
    """
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    
    # Load SirenLedger configuration
    try:
        config_manager = ConfigManager(config_path)
        app.config['SIREN_CONFIG'] = config_manager.config
        database = Database(config_manager.config.database)
        app.config['DATABASE'] = database
        
        # Initialize WebSocket manager
        websocket_manager = WebSocketManager(socketio, database)
        app.config['WEBSOCKET_MANAGER'] = websocket_manager
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        raise
    
    # Flask configuration
    app.config.update(
        SECRET_KEY='dev-key-change-in-production',  # TODO: Make configurable
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True,
    )
    
    # Register blueprints
    app.register_blueprint(create_api_blueprint(), url_prefix='/api/v1')
    app.register_blueprint(create_dashboard_blueprint(), url_prefix='/')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Not found'}), 404
        return render_template('error.html', error='Page not found'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('error.html', error='Internal server error'), 500
    
    # CLI commands
    @app.cli.command()
    @click.option('--host', default='127.0.0.1', help='Host to bind to')
    @click.option('--port', default=5555, help='Port to bind to')
    @click.option('--debug', is_flag=True, help='Enable debug mode')
    @with_appcontext
    def run_server(host: str, port: int, debug: bool) -> None:
        """Run the Flask development server."""
        app.run(host=host, port=port, debug=debug)
    
    @app.cli.command()
    @with_appcontext
    def init_db() -> None:
        """Initialize the database."""
        try:
            database = app.config['DATABASE']
            database.create_tables()
            click.echo('✓ Database initialized')
        except Exception as e:
            click.echo(f'Database initialization failed: {e}', err=True)
    
    return app, socketio


def main() -> None:
    """Main entry point for web application."""
    import sys
    from pathlib import Path
    
    # Parse command line arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    
    # Create and run app with WebSocket support
    app, socketio = create_app(config_path)
    socketio.run(app, host='0.0.0.0', port=5555, debug=True)


if __name__ == '__main__':
    main()