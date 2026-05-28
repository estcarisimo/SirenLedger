"""Dashboard blueprint with Plotly visualizations."""

import json
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

import plotly.graph_objects as go
import plotly.express as px
from flask import Blueprint, render_template_string, current_app, request, jsonify
from plotly.utils import PlotlyJSONEncoder

from ..storage import Database

logger = logging.getLogger(__name__)


def create_dashboard_blueprint() -> Blueprint:
    """Create dashboard blueprint with visualization routes.
    
    Returns
    -------
    Blueprint
        Flask blueprint for dashboard routes.
    """
    dashboard = Blueprint('dashboard', __name__)
    
    # HTML template for the dashboard
    DASHBOARD_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SirenLedger Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .chart-container { margin: 20px 0; }
            .stat-card { background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 10px 0; }
            .stat-value { font-size: 2em; font-weight: bold; color: #0066cc; }
            .stat-label { color: #666; }
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="row">
                <div class="col-12">
                    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                        <div class="container-fluid">
                            <a class="navbar-brand" href="#">SirenLedger Dashboard</a>
                            <div class="navbar-nav ms-auto">
                                <span class="navbar-text">Emergency Vehicle Siren Monitoring</span>
                            </div>
                        </div>
                    </nav>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <h2>Overview</h2>
                </div>
            </div>
            
            <!-- Summary Statistics -->
            <div class="row">
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value" id="total-events">-</div>
                        <div class="stat-label">Total Events (30 days)</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value" id="total-duration">-</div>
                        <div class="stat-label">Total Duration (hours)</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value" id="avg-per-day">-</div>
                        <div class="stat-label">Avg Events/Day</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value" id="last-activity">-</div>
                        <div class="stat-label">Last Activity</div>
                    </div>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="row mt-4">
                <div class="col-md-8">
                    <div class="chart-container">
                        <h4>Daily Siren Counts (Last 30 Days)</h4>
                        <div id="daily-counts-chart"></div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="chart-container">
                        <h4>Hourly Distribution</h4>
                        <div id="hourly-chart"></div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Daily Duration (Minutes)</h4>
                        <div id="duration-chart"></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Recent Events Timeline</h4>
                        <div id="timeline-chart"></div>
                    </div>
                </div>
            </div>
            
            <!-- Data Table -->
            <div class="row mt-4">
                <div class="col-12">
                    <h4>Recent Activity</h4>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Events</th>
                                    <th>Detections</th>
                                    <th>Duration (min)</th>
                                    <th>Longest Event (min)</th>
                                    <th>Avg Confidence</th>
                                </tr>
                            </thead>
                            <tbody id="recent-data-table">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Fetch and display data
            async function loadDashboard() {
                try {
                    // Load summary statistics
                    const summaryResponse = await fetch('/api/v1/stats/summary');
                    const summaryData = await summaryResponse.json();
                    
                    document.getElementById('total-events').textContent = summaryData.total_events;
                    document.getElementById('total-duration').textContent = summaryData.total_duration_hours;
                    document.getElementById('avg-per-day').textContent = summaryData.avg_events_per_day;
                    document.getElementById('last-activity').textContent = 
                        summaryData.last_activity_date ? new Date(summaryData.last_activity_date).toLocaleDateString() : 'None';
                    
                    // Load daily reports
                    const reportsResponse = await fetch('/api/v1/reports/daily?limit=30');
                    const reportsData = await reportsResponse.json();
                    
                    // Create daily counts chart
                    createDailyCountsChart(reportsData.reports);
                    
                    // Create duration chart
                    createDurationChart(reportsData.reports);
                    
                    // Populate data table
                    populateDataTable(reportsData.reports);
                    
                    // Load hourly distribution
                    const hourlyResponse = await fetch('/api/v1/stats/hourly?days=30');
                    const hourlyData = await hourlyResponse.json();
                    createHourlyChart(hourlyData.hourly_distribution);
                    
                    // Load recent events for timeline
                    const eventsResponse = await fetch('/api/v1/events?limit=50');
                    const eventsData = await eventsResponse.json();
                    createTimelineChart(eventsData.events);
                    
                } catch (error) {
                    console.error('Error loading dashboard:', error);
                }
            }
            
            function createDailyCountsChart(reports) {
                const dates = reports.map(r => r.date).reverse();
                const counts = reports.map(r => r.total_events).reverse();
                
                const trace = {
                    x: dates,
                    y: counts,
                    type: 'bar',
                    marker: { color: 'rgba(0, 102, 204, 0.7)' },
                    name: 'Events'
                };
                
                const layout = {
                    title: '',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Number of Events' },
                    margin: { t: 20 }
                };
                
                Plotly.newPlot('daily-counts-chart', [trace], layout, {responsive: true});
            }
            
            function createDurationChart(reports) {
                const dates = reports.map(r => r.date).reverse();
                const durations = reports.map(r => r.total_duration_minutes).reverse();
                
                const trace = {
                    x: dates,
                    y: durations,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: 'rgba(255, 99, 132, 1)' },
                    marker: { color: 'rgba(255, 99, 132, 0.7)' },
                    name: 'Duration'
                };
                
                const layout = {
                    title: '',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Duration (minutes)' },
                    margin: { t: 20 }
                };
                
                Plotly.newPlot('duration-chart', [trace], layout, {responsive: true});
            }
            
            function createHourlyChart(hourlyData) {
                const trace = {
                    x: hourlyData.hours,
                    y: hourlyData.totals,
                    type: 'bar',
                    marker: { color: 'rgba(75, 192, 192, 0.7)' },
                    name: 'Events'
                };
                
                const layout = {
                    title: '',
                    xaxis: { title: 'Hour of Day' },
                    yaxis: { title: 'Total Events' },
                    margin: { t: 20 }
                };
                
                Plotly.newPlot('hourly-chart', [trace], layout, {responsive: true});
            }
            
            function createTimelineChart(events) {
                if (events.length === 0) {
                    document.getElementById('timeline-chart').innerHTML = '<p>No recent events</p>';
                    return;
                }
                
                const trace = {
                    x: events.map(e => e.start_time),
                    y: events.map(e => e.max_confidence),
                    mode: 'markers',
                    marker: {
                        size: events.map(e => Math.max(6, e.detection_count * 2)),
                        color: events.map(e => e.max_confidence),
                        colorscale: 'Viridis',
                        showscale: true,
                        colorbar: { title: 'Confidence' }
                    },
                    text: events.map(e => `${e.dominant_class}<br>Confidence: ${e.max_confidence.toFixed(2)}<br>Detections: ${e.detection_count}`),
                    hovertemplate: '%{text}<extra></extra>',
                    name: 'Events'
                };
                
                const layout = {
                    title: '',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Confidence', range: [0, 1] },
                    margin: { t: 20 }
                };
                
                Plotly.newPlot('timeline-chart', [trace], layout, {responsive: true});
            }
            
            function populateDataTable(reports) {
                const tbody = document.getElementById('recent-data-table');
                tbody.innerHTML = '';
                
                reports.slice(0, 10).forEach(report => {
                    const row = tbody.insertRow();
                    row.insertCell(0).textContent = new Date(report.date).toLocaleDateString();
                    row.insertCell(1).textContent = report.total_events;
                    row.insertCell(2).textContent = report.total_detections;
                    row.insertCell(3).textContent = report.total_duration_minutes.toFixed(1);
                    row.insertCell(4).textContent = report.longest_event_duration.toFixed(1);
                    row.insertCell(5).textContent = report.avg_confidence.toFixed(2);
                });
            }
            
            // Load dashboard on page load
            loadDashboard();
            
            // Refresh every 30 seconds
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """
    
    @dashboard.route('/')
    def index():
        """Main dashboard page."""
        # Check if WebSocket support is available
        if 'WEBSOCKET_MANAGER' in current_app.config:
            # Use WebSocket-enabled template
            from flask import render_template
            return render_template('dashboard_websocket.html')
        else:
            # Fall back to polling-based template
            return render_template_string(DASHBOARD_TEMPLATE)
    
    @dashboard.route('/charts/daily-counts')
    def daily_counts_chart():
        """Generate daily counts chart data."""
        try:
            days = request.args.get('days', 30, type=int)
            
            database = current_app.config['DATABASE']
            reports = database.get_daily_reports(limit=days)
            
            if not reports:
                return jsonify({'error': 'No data available'}), 404
            
            # Prepare data for Plotly
            dates = [report.date.isoformat() for report in reversed(reports)]
            counts = [report.total_events for report in reversed(reports)]
            durations = [report.total_duration_minutes for report in reversed(reports)]
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add events bar chart
            fig.add_trace(go.Bar(
                x=dates,
                y=counts,
                name='Events',
                marker_color='rgba(0, 102, 204, 0.7)',
                yaxis='y'
            ))
            
            # Add duration line chart on secondary axis
            fig.add_trace(go.Scatter(
                x=dates,
                y=durations,
                mode='lines+markers',
                name='Duration (min)',
                line=dict(color='rgba(255, 99, 132, 1)'),
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Daily Siren Activity (Last {days} Days)',
                xaxis_title='Date',
                yaxis=dict(title='Number of Events', side='left'),
                yaxis2=dict(title='Duration (minutes)', side='right', overlaying='y'),
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Convert to JSON
            graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
            
            return jsonify({
                'chart': json.loads(graphJSON),
                'data_points': len(dates)
            })
            
        except Exception as e:
            logger.error(f"Failed to generate daily counts chart: {e}")
            return jsonify({'error': 'Chart generation failed'}), 500
    
    @dashboard.route('/charts/hourly-distribution')
    def hourly_distribution_chart():
        """Generate hourly distribution chart data."""
        try:
            days = request.args.get('days', 30, type=int)
            
            database = current_app.config['DATABASE']
            reports = database.get_daily_reports(limit=days)
            
            if not reports:
                return jsonify({'error': 'No data available'}), 404
            
            # Aggregate hourly data
            hourly_totals = [0] * 24
            total_days = len(reports)
            
            for report in reports:
                if report.events_by_hour and len(report.events_by_hour) == 24:
                    for hour, count in enumerate(report.events_by_hour):
                        hourly_totals[hour] += count
            
            # Create Plotly figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(range(24)),
                y=hourly_totals,
                name='Total Events',
                marker_color='rgba(75, 192, 192, 0.7)',
                text=[f'{h}:00' for h in range(24)],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'Hourly Distribution (Last {days} Days)',
                xaxis_title='Hour of Day',
                yaxis_title='Total Events',
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                template='plotly_white'
            )
            
            # Convert to JSON
            graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
            
            return jsonify({
                'chart': json.loads(graphJSON),
                'total_events': sum(hourly_totals),
                'days_analyzed': total_days
            })
            
        except Exception as e:
            logger.error(f"Failed to generate hourly distribution chart: {e}")
            return jsonify({'error': 'Chart generation failed'}), 500
    
    @dashboard.route('/charts/confidence-timeline')
    def confidence_timeline_chart():
        """Generate confidence timeline chart data."""
        try:
            days = request.args.get('days', 7, type=int)
            limit = request.args.get('limit', 100, type=int)
            
            # Get recent events
            database = current_app.config['DATABASE']
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            events = database.get_events_by_date_range(start_date, end_date)
            
            if not events:
                return jsonify({'error': 'No events in date range'}), 404
            
            # Limit number of events for performance
            if len(events) > limit:
                events = events[-limit:]
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Color map for different classes
            class_colors = {
                'siren': 'red',
                'police car': 'blue', 
                'ambulance': 'green',
                'fire engine': 'orange',
                'fire truck': 'orange'
            }
            
            # Group events by class
            for class_name in set(event.dominant_class for event in events):
                class_events = [e for e in events if e.dominant_class == class_name]
                
                fig.add_trace(go.Scatter(
                    x=[e.start_time for e in class_events],
                    y=[e.max_confidence for e in class_events],
                    mode='markers',
                    name=class_name,
                    marker=dict(
                        size=[max(6, e.detection_count * 2) for e in class_events],
                        color=class_colors.get(class_name.lower(), 'gray'),
                        opacity=0.7
                    ),
                    text=[f'Detections: {e.detection_count}<br>Duration: {e.duration or 0:.1f}s' 
                          for e in class_events],
                    hovertemplate='%{fullData.name}<br>Time: %{x}<br>Confidence: %{y:.2f}<br>%{text}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f'Siren Detection Timeline (Last {days} Days)',
                xaxis_title='Time',
                yaxis_title='Detection Confidence',
                yaxis=dict(range=[0, 1]),
                hovermode='closest',
                template='plotly_white'
            )
            
            # Convert to JSON
            graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
            
            return jsonify({
                'chart': json.loads(graphJSON),
                'events_shown': len(events),
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to generate confidence timeline chart: {e}")
            return jsonify({'error': 'Chart generation failed'}), 500
    
    return dashboard