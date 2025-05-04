"""
벡터 데이터베이스 저장소 클래스
장기 메모리를 위한 의미 기반 검색
"""
import chromadb
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from models.memory_entry import MemoryEntry


class VectorStorage:
    """
    벡터 저장소 - 의미 기반 메모리 검색
    
    특징:
    - 임베딩 기반 유사도 검색
    - 효율적인 근사 최근접 이웃 탐색
    - 대규모 데이터 처리 최적화
    """
    
    def __init__(self, db_path: str, collection_name: str, embedding_model: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 임베딩 모델 초기화
        self.embedder = SentenceTransformer(embedding_model)
    
    async def save(self, memory: MemoryEntry) -> None:
        """메모리 저장"""
        # 텍스트 임베딩 생성
        text = self._extract_text(memory)
        embedding = self.embedder.encode(text).tolist()
        
        # 메타데이터 준비
        metadata = {
            'concepts': json.dumps(memory.concepts),
            'importance': memory.importance,
            'emotional_weight': memory.emotional_weight,
            'creation_time': memory.creation_time.isoformat(),
            'tier': memory.tier.value
        }
        
        # 저장
        self.collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[json.dumps(memory.content)],
            metadatas=[metadata]
        )
    
    async def search_by_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """텍스트 검색"""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._parse_results(results)
    
    async def search_by_concepts(self, concepts: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """개념으로 검색"""
        # 개념을 문장으로 결합
        query_text = " ".join(concepts)
        return await self.search_by_text(query_text, top_k)
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """특정 메모리 조회"""
        results = self.collection.get(ids=[memory_id], include=['documents', 'metadatas'])
        
        if results['ids']:
            data = {
                'id': results['ids'][0],
                'content': json.loads(results['documents'][0]),
                'concepts': json.loads(results['metadatas'][0]['concepts']),
                'importance': results['metadatas'][0]['importance'],
                'emotional_weight': results['metadatas'][0]['emotional_weight'],
                'creation_time': results['metadatas'][0]['creation_time'],
                'tier': results['metadatas'][0]['tier']
            }
            return MemoryEntry.from_dict(data)
        
        return None
    
    async def delete(self, memory_id: str) -> None:
        """메모리 삭제"""
        self.collection.delete(ids=[memory_id])
    
    def _extract_text(self, memory: MemoryEntry) -> str:
        """메모리에서 검색 가능한 텍스트 추출"""
        content = memory.content
        text_parts = []
        
        # 사용자 입력과 응답 추가
        if 'user' in content:
            text_parts.append(content['user'])
        if 'assistant' in content:
            text_parts.append(content['assistant'])
        
        # 개념 추가 (검색 정확도 향상)
        text_parts.extend(memory.concepts)
        
        return " ".join(text_parts)
    
    def _parse_results(self, results: Dict) -> List[Dict[str, Any]]:
        """검색 결과 파싱"""
        parsed_results = []
        
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                parsed_results.append({
                    'id': results['ids'][0][i],
                    'content': json.loads(results['documents'][0][i]),
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # 유사도 계산
                })
        
        return parsed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        return {
            'total_memories': self.collection.count(),
            'collection_name': self.collection.name,
            'metadata': self.collection.metadata
        }
