"""Command-line interface for SirenLedger."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from ..config import ConfigManager
from ..service import SirenDetectorService
from ..storage import Database
from ..audio import AudioProcessor


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Parameters
    ----------
    level : str
        Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    default='INFO',
    help='Logging level'
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], log_level: str) -> None:
    """SirenLedger: Emergency vehicle siren detection and logging system."""
    setup_logging(log_level)
    
    try:
        config_manager = ConfigManager(config)
        ctx.ensure_object(dict)
        ctx.obj['config_manager'] = config_manager
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Run the siren detection service."""
    config_manager = ctx.obj['config_manager']
    
    try:
        service = SirenDetectorService(config_manager)
        service.run()
    except Exception as e:
        click.echo(f"Service error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current system status."""
    config_manager = ctx.obj['config_manager']
    
    try:
        # Display configuration summary
        click.echo("=== SirenLedger Status ===")
        click.echo()
        click.echo(config_manager.get_summary())
        
        # Test database connection
        database = Database(config_manager.config.database)
        click.echo("Database: ✓ Connected")
        
        # Get recent statistics
        reports = database.get_daily_reports(limit=7)
        if reports:
            click.echo("\nRecent Activity (last 7 days):")
            for report in reports:
                click.echo(f"  {report.date}: {report.total_events} events, "
                          f"{report.total_duration_minutes:.1f} minutes")
        else:
            click.echo("No recent activity data")
            
    except Exception as e:
        click.echo(f"Status check failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--device-id', '-d',
    type=int,
    help='Audio device ID to test'
)
@click.option(
    '--duration', '-t',
    type=float,
    default=5.0,
    help='Test duration in seconds'
)
def test_audio(device_id: Optional[int], duration: float) -> None:
    """Test audio device functionality."""
    click.echo("=== Audio Device Test ===")
    
    # List available devices
    click.echo("\nAvailable audio devices:")
    AudioProcessor.list_audio_devices()
    
    # Test specified or default device
    click.echo(f"\nTesting device {device_id or 'default'} for {duration} seconds...")
    
    success = AudioProcessor.test_audio_device(device_id, duration)
    
    if success:
        click.echo("✓ Audio device test passed")
    else:
        click.echo("✗ Audio device test failed", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default=Path('siren_config.yaml'),
    help='Output configuration file path'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['yaml', 'json']),
    default='yaml',
    help='Configuration file format'
)
def init_config(output: Path, format: str) -> None:
    """Create a default configuration file."""
    try:
        if output.exists():
            if not click.confirm(f"Configuration file {output} already exists. Overwrite?"):
                return
        
        ConfigManager.create_default_config(output)
        click.echo(f"✓ Default configuration created: {output}")
        click.echo("\nEdit the configuration file to customize settings, then run:")
        click.echo(f"  siren-counter -c {output} run")
        
    except Exception as e:
        click.echo(f"Failed to create configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--csv-file', '-f',
    type=click.Path(exists=True, path_type=Path),
    default=Path('siren_daily_counts.csv'),
    help='Path to legacy CSV file'
)
@click.pass_context
def migrate_csv(ctx: click.Context, csv_file: Path) -> None:
    """Migrate data from legacy CSV format."""
    config_manager = ctx.obj['config_manager']
    
    try:
        database = Database(config_manager.config.database)
        
        if not csv_file.exists():
            click.echo(f"CSV file not found: {csv_file}", err=True)
            sys.exit(1)
        
        click.echo(f"Migrating data from {csv_file}...")
        count = database.migrate_csv_data(csv_file)
        
        click.echo(f"✓ Migrated {count} daily records")
        
        if count > 0:
            click.echo("Run 'siren-counter status' to view migrated data")
        
    except Exception as e:
        click.echo(f"Migration failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--days', '-d',
    type=int,
    default=365,
    help='Number of days of data to keep'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be deleted without actually deleting'
)
@click.pass_context
def cleanup(ctx: click.Context, days: int, dry_run: bool) -> None:
    """Clean up old data beyond retention period."""
    config_manager = ctx.obj['config_manager']
    
    try:
        database = Database(config_manager.config.database)
        
        if dry_run:
            click.echo(f"DRY RUN: Would delete data older than {days} days")
            # Could implement a count query here
            return
        
        if not click.confirm(f"Delete all data older than {days} days?"):
            return
        
        deleted_events, deleted_detections = database.cleanup_old_data(days)
        
        click.echo(f"✓ Cleaned up {deleted_events} events and {deleted_detections} detections")
        
    except Exception as e:
        click.echo(f"Cleanup failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--start-date', '-s',
    type=click.DateTime(formats=['%Y-%m-%d']),
    help='Start date (YYYY-MM-DD)'
)
@click.option(
    '--end-date', '-e', 
    type=click.DateTime(formats=['%Y-%m-%d']),
    help='End date (YYYY-MM-DD)'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['table', 'json', 'csv']),
    default='table',
    help='Output format'
)
@click.pass_context
def report(ctx: click.Context, start_date, end_date, format: str) -> None:
    """Generate activity reports."""
    config_manager = ctx.obj['config_manager']
    
    try:
        database = Database(config_manager.config.database)
        
        # Convert datetime to date if provided
        start = start_date.date() if start_date else None
        end = end_date.date() if end_date else None
        
        reports = database.get_daily_reports(start, end)
        
        if not reports:
            click.echo("No data found for the specified date range")
            return
        
        if format == 'table':
            _display_table_report(reports)
        elif format == 'json':
            _display_json_report(reports)
        elif format == 'csv':
            _display_csv_report(reports)
            
    except Exception as e:
        click.echo(f"Report generation failed: {e}", err=True)
        sys.exit(1)


def _display_table_report(reports) -> None:
    """Display reports in table format."""
    click.echo("Date       | Events | Duration (min) | Avg Confidence")
    click.echo("-" * 55)
    
    for report in reports:
        click.echo(f"{report.date} |   {report.total_events:4d} |        {report.total_duration_minutes:6.1f} |        {report.avg_confidence:6.2f}")


def _display_json_report(reports) -> None:
    """Display reports in JSON format."""
    import json
    
    data = [report.dict() for report in reports]
    click.echo(json.dumps(data, indent=2, default=str))


def _display_csv_report(reports) -> None:
    """Display reports in CSV format."""
    click.echo("date,events,detections,duration_minutes,longest_event,avg_confidence")
    
    for report in reports:
        click.echo(f"{report.date},{report.total_events},{report.total_detections},"
                  f"{report.total_duration_minutes},{report.longest_event_duration},"
                  f"{report.avg_confidence}")


def main() -> None:
    """Main entry point for CLI."""
    cli()