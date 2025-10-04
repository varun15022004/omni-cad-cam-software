"""
PLM (Product Lifecycle Management) System for OmniCAD

Provides comprehensive PLM/PDM capabilities including:
- Version control and revision management
- Change management and approval workflows
- Data lifecycle management
- Collaboration tools and user management
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from ..utils.logger import OmniLogger
from ..utils.event_system import EventSystem


class DocumentState(Enum):
    """Document lifecycle states"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    RELEASED = "released"
    OBSOLETE = "obsolete"


class ChangeType(Enum):
    """Types of changes"""
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class User:
    """System user"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    role: str = "user"  # user, engineer, manager, admin
    permissions: List[str] = field(default_factory=list)


@dataclass
class DocumentRevision:
    """Document revision"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    version: str = "1.0"
    state: DocumentState = DocumentState.DRAFT
    created_by: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    comment: str = ""
    file_path: str = ""
    checksum: str = ""


@dataclass
class ChangeRequest:
    """Change request/ECO"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    change_type: ChangeType = ChangeType.MINOR
    requested_by: str = ""
    requested_date: str = field(default_factory=lambda: datetime.now().isoformat())
    affected_documents: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, approved, rejected, implemented
    approvers: List[str] = field(default_factory=list)


class PLMManager:
    """Main PLM/PDM manager"""
    
    def __init__(self):
        self.logger = OmniLogger("PLMManager")
        self.event_system = EventSystem()
        
        # PLM data
        self.users: Dict[str, User] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.revisions: Dict[str, DocumentRevision] = {}
        self.change_requests: Dict[str, ChangeRequest] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
        # Current user
        self.current_user_id: Optional[str] = None
        
        # Initialize default user
        self._create_default_user()
    
    async def initialize(self, app_context):
        """Initialize PLM manager"""
        self.app_context = app_context
        self.logger.info("PLM Manager initialized")
    
    def _create_default_user(self):
        """Create default admin user"""
        admin_user = User(
            username="admin",
            email="admin@omnicad.com",
            role="admin",
            permissions=["read", "write", "admin", "approve"]
        )
        self.users[admin_user.id] = admin_user
        self.current_user_id = admin_user.id
    
    def create_document(self, name: str, description: str = "", project_data: Dict[str, Any] = None) -> str:
        """Create new document"""
        document_id = str(uuid.uuid4())
        
        document = {
            'id': document_id,
            'name': name,
            'description': description,
            'created_by': self.current_user_id,
            'created_date': datetime.now().isoformat(),
            'current_revision_id': None,
            'all_revisions': [],
            'metadata': {
                'part_number': f'PN-{document_id[:8]}',
                'category': 'General',
                'tags': []
            }
        }
        
        self.documents[document_id] = document
        
        # Create initial revision
        if project_data:
            revision_id = self.create_revision(document_id, "1.0", "Initial version", project_data)
            document['current_revision_id'] = revision_id
        
        self.event_system.emit('document_created', {
            'document_id': document_id,
            'name': name
        })
        
        return document_id
    
    def create_revision(self, document_id: str, version: str, comment: str, data: Dict[str, Any]) -> str:
        """Create new document revision"""
        document = self.documents.get(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        revision = DocumentRevision(
            document_id=document_id,
            version=version,
            created_by=self.current_user_id or "",
            comment=comment,
            file_path=f"rev_{version}_{document_id}.json",
            checksum=self._calculate_checksum(data)
        )
        
        self.revisions[revision.id] = revision
        document['all_revisions'].append(revision.id)
        document['current_revision_id'] = revision.id
        
        self.event_system.emit('revision_created', {
            'document_id': document_id,
            'revision_id': revision.id,
            'version': version
        })
        
        return revision.id
    
    def create_change_request(self, title: str, description: str, affected_docs: List[str], change_type: ChangeType = ChangeType.MINOR) -> str:
        """Create change request"""
        change_request = ChangeRequest(
            title=title,
            description=description,
            change_type=change_type,
            requested_by=self.current_user_id or "",
            affected_documents=affected_docs
        )
        
        self.change_requests[change_request.id] = change_request
        
        self.event_system.emit('change_request_created', {
            'change_request_id': change_request.id,
            'title': title
        })
        
        return change_request.id
    
    def approve_change_request(self, change_request_id: str, approver_id: str) -> bool:
        """Approve change request"""
        change_request = self.change_requests.get(change_request_id)
        if not change_request:
            return False
        
        if approver_id not in change_request.approvers:
            change_request.approvers.append(approver_id)
        
        # Simple approval logic - if any approver, mark as approved
        if change_request.approvers:
            change_request.status = "approved"
        
        self.event_system.emit('change_request_approved', {
            'change_request_id': change_request_id,
            'approver_id': approver_id
        })
        
        return True
    
    def get_document_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Get document revision history"""
        document = self.documents.get(document_id)
        if not document:
            return []
        
        history = []
        for revision_id in document['all_revisions']:
            revision = self.revisions.get(revision_id)
            if revision:
                user = self.users.get(revision.created_by, User())
                history.append({
                    'revision_id': revision.id,
                    'version': revision.version,
                    'state': revision.state.value,
                    'created_by': user.username,
                    'created_date': revision.created_date,
                    'comment': revision.comment
                })
        
        return sorted(history, key=lambda x: x['created_date'], reverse=True)
    
    def transition_document_state(self, document_id: str, new_state: DocumentState) -> bool:
        """Transition document to new state"""
        document = self.documents.get(document_id)
        if not document:
            return False
        
        current_revision_id = document.get('current_revision_id')
        if not current_revision_id:
            return False
        
        revision = self.revisions.get(current_revision_id)
        if not revision:
            return False
        
        old_state = revision.state
        revision.state = new_state
        
        self.event_system.emit('document_state_changed', {
            'document_id': document_id,
            'old_state': old_state.value,
            'new_state': new_state.value
        })
        
        return True
    
    def search_documents(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search documents"""
        results = []
        filters = filters or {}
        
        for document in self.documents.values():
            # Simple text search
            if query.lower() in document['name'].lower() or query.lower() in document['description'].lower():
                # Apply filters
                if self._document_matches_filters(document, filters):
                    current_revision = None
                    if document.get('current_revision_id'):
                        current_revision = self.revisions.get(document['current_revision_id'])
                    
                    results.append({
                        'document_id': document['id'],
                        'name': document['name'],
                        'description': document['description'],
                        'part_number': document['metadata']['part_number'],
                        'current_version': current_revision.version if current_revision else "N/A",
                        'state': current_revision.state.value if current_revision else "unknown",
                        'created_date': document['created_date']
                    })
        
        return results
    
    def _document_matches_filters(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches filters"""
        # Filter by state
        if 'state' in filters:
            current_revision = None
            if document.get('current_revision_id'):
                current_revision = self.revisions.get(document['current_revision_id'])
            if not current_revision or current_revision.state.value != filters['state']:
                return False
        
        # Filter by category
        if 'category' in filters:
            if document['metadata']['category'] != filters['category']:
                return False
        
        return True
    
    def get_user_activity(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get user activity history"""
        # This would typically query an activity log
        # For now, return documents created by user
        activity = []
        
        for document in self.documents.values():
            if document['created_by'] == user_id:
                activity.append({
                    'action': 'created_document',
                    'document_name': document['name'],
                    'date': document['created_date']
                })
        
        for revision in self.revisions.values():
            if revision.created_by == user_id:
                document = self.documents.get(revision.document_id)
                if document:
                    activity.append({
                        'action': 'created_revision',
                        'document_name': document['name'],
                        'version': revision.version,
                        'date': revision.created_date
                    })
        
        return sorted(activity, key=lambda x: x['date'], reverse=True)
    
    def generate_bom_with_versions(self, document_id: str) -> Dict[str, Any]:
        """Generate BOM with version information"""
        # This would integrate with the assembly BOM but add PLM data
        bom = {
            'document_id': document_id,
            'generated_date': datetime.now().isoformat(),
            'items': []
        }
        
        # Placeholder BOM items with PLM data
        for i in range(5):
            bom['items'].append({
                'item_number': i + 1,
                'part_number': f'PN-{i+1:03d}',
                'description': f'Component {i+1}',
                'version': '1.0',
                'state': 'released',
                'quantity': 1
            })
        
        return bom
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate data checksum"""
        import hashlib
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get PLM statistics"""
        return {
            'total_users': len(self.users),
            'total_documents': len(self.documents),
            'total_revisions': len(self.revisions),
            'pending_change_requests': sum(1 for cr in self.change_requests.values() if cr.status == 'pending'),
            'document_states': {
                state.value: sum(1 for r in self.revisions.values() if r.state == state)
                for state in DocumentState
            }
        }
    
    def shutdown(self):
        """Shutdown PLM manager"""
        self.logger.info("PLM Manager shutdown")