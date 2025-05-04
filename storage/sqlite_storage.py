"""
SQLite 저장소 클래스
단기 메모리 및 관계 데이터 관리
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from models.memory_entry import MemoryEntry
from models.enums import MemoryTier


class SQLiteStorage:
    """
    SQLite 단기 메모리 저장소
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 메모리 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    concepts TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    emotional_weight REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    tier TEXT NOT NULL,
                    metadata TEXT,
                    last_accessed TIMESTAMP,
                    creation_time TIMESTAMP NOT NULL
                )
            ''')
            
            # 개념 테이블 생성 (추가!)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            ''')
                    # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts ON memories(concepts)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_creation_time ON memories(creation_time)')

            # 개념 테이블 인덱스
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_text ON concepts(concept)')

            conn.commit()
                
    
    async def save(self, memory: MemoryEntry) -> None:
        """메모리 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.id,
                json.dumps(memory.content),
                json.dumps(memory.concepts),
                memory.importance,
                memory.emotional_weight,
                memory.access_count,
                memory.tier.value,
                json.dumps(memory.metadata),
                memory.last_accessed.isoformat() if memory.last_accessed else None,
                memory.creation_time.isoformat()
            ))
            
            conn.commit()
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """메모리 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_memory(row)
            return None
    
    async def find_by_concepts(self, concepts: List[str], limit: int = 10) -> List[MemoryEntry]:
        """개념으로 메모리 검색"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 간단한 LIKE 쿼리로 변경
            where_clauses = []
            params = []
            
            for concept in concepts:
                where_clauses.append("concepts LIKE ?")
                params.append(f'%{concept}%')
            
            if not where_clauses:
                return []
            
            query = f'''
                SELECT * FROM memories 
                WHERE {" OR ".join(where_clauses)}
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            '''
            
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    async def update_access(self, memory_id: str) -> None:
        """접근 정보 업데이트"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE memories 
                SET access_count = access_count + 1,
                    last_accessed = ? 
                WHERE id = ?
            ''', (datetime.now().isoformat(), memory_id))
            
            conn.commit()
    
    async def find_by_tier(self, tier: MemoryTier) -> List[MemoryEntry]:
        """티어별 메모리 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memories WHERE tier = ?', (tier.value,))
            rows = cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    async def delete(self, memory_id: str) -> None:
        """메모리 삭제"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            conn.commit()
    
    def _row_to_memory(self, row: tuple) -> MemoryEntry:
        """데이터베이스 행을 MemoryEntry로 변환"""
        return MemoryEntry(
            id=row[0],
            content=json.loads(row[1]),
            concepts=json.loads(row[2]),
            importance=row[3],
            emotional_weight=row[4],
            access_count=row[5],
            tier=MemoryTier(row[6]),
            metadata=json.loads(row[7]) if row[7] else {},
            last_accessed=datetime.fromisoformat(row[8]) if row[8] else None,
            creation_time=datetime.fromisoformat(row[9])
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 총 메모리 수
            cursor.execute('SELECT COUNT(*) FROM memories')
            total_count = cursor.fetchone()[0]
            
            # 티어별 분포
            cursor.execute('''
                SELECT tier, COUNT(*) 
                FROM memories 
                GROUP BY tier
            ''')
            tier_distribution = dict(cursor.fetchall())
            
            # 평균 중요도
            cursor.execute('SELECT AVG(importance) FROM memories')
            avg_importance = cursor.fetchone()[0] or 0.0
            
            return {
                'total_memories': total_count,
                'tier_distribution': tier_distribution,
                'average_importance': avg_importance
            }

