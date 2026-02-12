#!/usr/bin/env python3
# modules/reports.py - Generate system reports
import sqlite3
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from config import DB_PATH, LOGS_DIR

class ReportGenerator:
    def __init__(self):
        self.db_path = str(DB_PATH)
        self.reports_dir = LOGS_DIR / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def generate_daily_report(self, date=None, output_format='json'):
        """Generate daily report of all batches"""
        if date is None:
            date = datetime.now().date()
        
        batches, stats = self._fetch_report_data(date)
        report = self._build_report_dict(date, batches, stats)
        
        filename = f"daily_report_{date.strftime('%Y%m%d')}"
        if output_format == 'json':
            filepath = self._save_report_json(report, filename)
        elif output_format == 'csv':
            filepath = self._save_report_csv(report, filename)
        
        print(f"Report saved to: {filepath}")
        return report

    def _fetch_report_data(self, date):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            # Get batch data
            cur.execute('''
                SELECT 
                    session_id, batch_status, decodedqrcodes, target_keg_count,
                    beer_type, session_timestamp, processing_time,
                    advanced_detection_used, api_attempts, last_error
                FROM detection_sessions 
                WHERE date(session_timestamp) = ?
                ORDER BY session_timestamp DESC
            ''', (date.strftime('%Y-%m-%d'),))
            batches = cur.fetchall()
            
            # Get summary statistics
            cur.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN batch_status = 'api_sent' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN batch_status = 'api_failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN batch_status = 'duplicate' THEN 1 ELSE 0 END) as duplicates,
                    AVG(processing_time) as avg_time,
                    SUM(decodedqrcodes) as total_qrs
                FROM detection_sessions 
                WHERE date(session_timestamp) = ?
            ''', (date.strftime('%Y-%m-%d'),))
            stats = cur.fetchone()
            
            return batches, stats
        finally:
            conn.close()

    def _build_report_dict(self, date, batches, stats):
        report = {
            'date': str(date),
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_batches': stats[0] or 0,
                'successful': stats[1] or 0,
                'failed': stats[2] or 0,
                'duplicates': stats[3] or 0,
                'success_rate': round((stats[1] / stats[0] * 100) if stats[0] and stats[0] > 0 else 0, 1),
                'avg_processing_time': round(stats[4] or 0, 2),
                'total_qr_codes': stats[5] or 0
            },
            'batches': []
        }
        
        for batch in batches:
            report['batches'].append({
                'id': batch[0],
                'status': batch[1],
                'qr_count': batch[2] or 0,
                'target_count': batch[3] or 6,
                'beer_type': batch[4] or 'Unknown',
                'time': batch[5],
                'processing_seconds': round(batch[6] or 0, 2),
                'advanced_used': bool(batch[7]),
                'api_attempts': batch[8] or 0,
                'error': batch[9]
            })
        return report

    def _save_report_json(self, report, filename):
        filepath = self.reports_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return filepath

    def _save_report_csv(self, report, filename):
        filepath = self.reports_dir / f"{filename}.csv"
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Date', 'Batch ID', 'Status', 'QR Count', 'Target Count',
                           'Beer Type', 'Time', 'Processing Time', 'Advanced Used',
                           'API Attempts', 'Error'])
            # Write data
            for batch in report['batches']:
                writer.writerow([
                    batch['time'],
                    batch['id'],
                    batch['status'],
                    batch['qr_count'],
                    batch['target_count'],
                    batch['beer_type'],
                    batch['time'],
                    batch['processing_seconds'],
                    batch['advanced_used'],
                    batch['api_attempts'],
                    batch['error'] or ''
                ])
        return filepath
    
    def generate_operator_report(self):
        """Generate report for operator showing attention needed"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                session_id,
                batch_status,
                attention_reason,
                session_timestamp,
                decodedqrcodes,
                target_keg_count,
                beer_type,
                api_attempts,
                last_error
            FROM detection_sessions 
            WHERE require_attention = 1
            ORDER BY session_timestamp DESC
        ''')
        
        attention_batches = cur.fetchall()
        
        cur.execute('''
            SELECT COUNT(*) FROM retry_queue
        ''')
        retry_count = cur.fetchone()[0]
        
        conn.close()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'attention_count': len(attention_batches),
            'retry_queue_count': retry_count,
            'batches': []
        }
        
        for batch in attention_batches:
            report['batches'].append({
                'id': batch[0],
                'status': batch[1],
                'reason': batch[2],
                'timestamp': batch[3],
                'qr_count': batch[4],
                'target_count': batch[5],
                'beer_type': batch[6],
                'api_attempts': batch[7],
                'last_error': batch[8],
                'action_required': 'retry' if batch[1] == 'api_failed' else 'review'
            })
        
        # Save report
        filename = f"operator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Operator report saved to: {filepath}")
        return report
    
    def generate_performance_report(self, days=7):
        """Generate performance report over specified days"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        cur.execute('''
            SELECT 
                date(session_timestamp) as day,
                COUNT(*) as total_batches,
                SUM(CASE WHEN batch_status = 'api_sent' THEN 1 ELSE 0 END) as successful,
                AVG(processing_time) as avg_time,
                SUM(decodedqrcodes) as total_qrs,
                SUM(CASE WHEN advanced_detection_used = 1 THEN 1 ELSE 0 END) as advanced_used
            FROM detection_sessions 
            WHERE date(session_timestamp) BETWEEN ? AND ?
            GROUP BY date(session_timestamp)
            ORDER BY day DESC
        ''', (start_date, end_date))
        
        daily_stats = cur.fetchall()
        
        # Calculate totals
        cur.execute('''
            SELECT 
                COUNT(*) as total_batches,
                SUM(CASE WHEN batch_status = 'api_sent' THEN 1 ELSE 0 END) as successful,
                AVG(processing_time) as avg_time,
                SUM(decodedqrcodes) as total_qrs
            FROM detection_sessions 
            WHERE date(session_timestamp) BETWEEN ? AND ?
        ''', (start_date, end_date))
        
        totals = cur.fetchone()
        conn.close()
        
        report = {
            'period': {
                'start': str(start_date),
                'end': str(end_date),
                'days': days
            },
            'generated_at': datetime.now().isoformat(),
            'totals': {
                'total_batches': totals[0] or 0,
                'successful': totals[1] or 0,
                'success_rate': round((totals[1] / totals[0] * 100) if totals[0] > 0 else 0, 1),
                'avg_processing_time': round(totals[2] or 0, 2),
                'total_qr_codes': totals[3] or 0
            },
            'daily': []
        }
        
        for day in daily_stats:
            report['daily'].append({
                'date': day[0],
                'total_batches': day[1],
                'successful': day[2],
                'success_rate': round((day[2] / day[1] * 100) if day[1] > 0 else 0, 1),
                'avg_processing_time': round(day[3] or 0, 2),
                'total_qr_codes': day[4],
                'advanced_used': day[5]
            })
        
        # Save report
        filename = f"performance_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.json"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Performance report saved to: {filepath}")
        return report

if __name__ == "__main__":
    # Example usage
    generator = ReportGenerator()
    
    # Generate today's report
    generator.generate_daily_report()
    
    # Generate operator report
    generator.generate_operator_report()
    
    # Generate weekly performance report
    generator.generate_performance_report(days=7)