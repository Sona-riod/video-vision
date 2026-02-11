#!/usr/bin/env python3
# modules/pallet_controller.py - Pallet lifecycle management
import sqlite3
import json
import hashlib
from typing import List, Optional
from datetime import datetime
from config import DB_PATH, PALLET_STATUS

class PalletController:
    def __init__(self):
        self.db_path = str(DB_PATH)
    
    def create_pallet(self, session_id: str, keg_type: str, keg_count: int, qr_codes: List[str],
                      beer_type: str, batch: str, filling_date: str) -> dict:
        """Create new pallet with duplicate prevention (Scene 8)"""
        from modules.database import DatabaseManager
        db = DatabaseManager()
        
        # Check for duplicate (not shipped)
        existing_pallet, status = db.check_pallet_duplicate(qr_codes)
        
        if existing_pallet and status != 'SHIPPED':
            return {
                'success': False,
                'error': 'PALLET_EXISTS',
                'message': f'Pallet {existing_pallet} already exists (status: {status})',
                'pallet_id': existing_pallet,
                'status': status
            }
        
    
        pallet_id = db.create_pallet_record(session_id, keg_type, keg_count, qr_codes)
        
        
        extended_data = {
            'beer_type': beer_type,
            'batch': batch,
            'filling_date': filling_date,
            'created_at': datetime.now().isoformat()
        }
        
       
        print(f"Pallet created with extended data: {extended_data}")
        
        return {
            'success': True,
            'pallet_id': pallet_id,
            'message': f'Pallet {pallet_id} created successfully - QR generation by cloud after batch send',
            'status': 'CREATED',
            'extended_data': extended_data  # Optional: include in response
        }
    
    def _generate_qr_data(self, pallet_id: str, keg_type: str, keg_count: int, qr_codes: List[str]) -> dict:
        """Generate QR code data for printing - kept for backward compatibility if needed"""
        # Create secure hash for data integrity
        data_string = f"{pallet_id}:{keg_type}:{keg_count}:{','.join(sorted(qr_codes))}"
        data_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        return {
            'pallet_id': pallet_id,
            'keg_type': keg_type,
            'keg_count': keg_count,
            'qr_codes': qr_codes,
            'qr_count': len(qr_codes),
            'timestamp': datetime.now().isoformat(),
            'data_hash': data_hash,
            'status': 'CREATED'
        }
    
    def update_pallet_status(self, pallet_id: str, new_status: str) -> dict:
        """Update pallet status (CREATED → SHIPPED → etc.)"""
        if new_status not in PALLET_STATUS:
            return {'success': False, 'error': 'INVALID_STATUS'}
        
        conn = sqlite3.connect(self.db_path, timeout=60)
        cur = conn.cursor()
        
        if new_status == 'SHIPPED':
            cur.execute('''
                UPDATE pallet_lifecycle 
                SET status = ?, shipped_at = datetime('now'), last_modified = datetime('now')
                WHERE pallet_id = ?
            ''', (new_status, pallet_id))
        else:
            cur.execute('''
                UPDATE pallet_lifecycle 
                SET status = ?, last_modified = datetime('now')
                WHERE pallet_id = ?
            ''', (new_status, pallet_id))
        
        conn.commit()
        conn.close()
        
        return {'success': True, 'pallet_id': pallet_id, 'status': new_status}
    
    def get_pallet_info(self, pallet_id: str) -> dict:
        """Get detailed pallet information"""
        conn = sqlite3.connect(self.db_path, timeout=60)
        cur = conn.cursor()
        
        cur.execute('''
            SELECT pl.*, ds.session_timestamp, ds.source_image
            FROM pallet_lifecycle pl
            LEFT JOIN detection_sessions ds ON pl.session_id = ds.session_id
            WHERE pl.pallet_id = ?
        ''', (pallet_id,))
        
        result = cur.fetchone()
        conn.close()
        
        if result:
            qr_data = None
            try:
                if result[10]:  # qr_data field
                    qr_data = json.loads(result[10])
            except Exception:
                pass
                
            return {
                'pallet_id': result[0],
                'session_id': result[1],
                'keg_type': result[2],
                'keg_count': result[3],
                'status': result[5],
                'created_at': result[6],
                'shipped_at': result[7],
                'last_modified': result[8],
                'qr_generated': bool(result[9]),
                'qr_data': qr_data,
                'session_timestamp': result[11],
                'source_image': result[12]
            }
        return None
    
    def check_duplicate_prevention(self, qr_codes: List[str]) -> dict:
        """Check if pallet can be created (duplicate prevention logic)"""
        from modules.database import DatabaseManager
        db = DatabaseManager()
        
        existing_pallet, status = db.check_pallet_duplicate(qr_codes)
        
        if existing_pallet:
            return {
                'can_create': False,
                'reason': 'DUPLICATE',
                'pallet_id': existing_pallet,
                'status': status,
                'message': f'Pallet {existing_pallet} already exists (status: {status})'
            }
        
        return {
            'can_create': True,
            'reason': 'NEW_PALLET',
            'message': 'No duplicate found, pallet can be created'
        }
    
    def create_pallet_with_metadata(self, session_id: str, keg_type: str, keg_count: int, 
                                    qr_codes: List[str], metadata: dict = None) -> dict:
        """Extended version that accepts metadata including beer_type, batch, filling_date"""
        # Extract metadata fields with defaults
        beer_type = metadata.get('beer_type', '') if metadata else ''
        batch = metadata.get('batch', '') if metadata else ''
        filling_date = metadata.get('filling_date', '') if metadata else ''
        
        # Use the main create_pallet method
        return self.create_pallet(
            session_id=session_id,
            keg_type=keg_type,
            keg_count=keg_count,
            qr_codes=qr_codes,
            beer_type=beer_type,
            batch=batch,
            filling_date=filling_date
        )

# For backward compatibility
def check_duplicate_pallet(qr_codes: List[str]) -> bool:
    """Simple duplicate check for existing code"""
    controller = PalletController()
    result = controller.check_duplicate_prevention(qr_codes)
    return not result['can_create']

if __name__ == "__main__":
    # Test the pallet controller
    controller = PalletController()
    
    # Test duplicate prevention
    test_qr_codes = ["QR1", "QR2", "QR3", "QR4", "QR5", "QR6"]
    result = controller.check_duplicate_prevention(test_qr_codes)
    print(f"Duplicate check: {result}")
    
    # Test create with metadata
    test_metadata = {
        'beer_type': 'IPA',
        'batch': 'BATCH-2024-001',
        'filling_date': '2024-01-15'
    }
    
    # Note: This would require an actual session_id and valid QR codes
    print(f"Test metadata: {test_metadata}")