"""
실시간 연관 네트워크 모듈
개념 간 연관 관계 관리 및 탐색
"""
import networkx as nx
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional

from models.enums import ConnectionType
from datetime import datetime, timedelta  # 추가된 import
from typing import Dict, List, Set, Tuple, Optional, Any  # Any 추가

class AssociationNetwork:
    """
    연관 네트워크 클래스
    
    기능:
    - 실시간 개념 연결 관리
    - 연관 강도 동적 조정
    - 확산 활성화 탐색
    """
    
    def __init__(self, min_strength: float = 0.1, decay_factor: float = 0.95):
        # 방향성 그래프 (개념 간 연관)
        self.graph = nx.DiGraph()
        
        # 설정값
        self.min_strength = min_strength
        self.decay_factor = decay_factor
        
        # 최근 활성화 기록
        self.recent_activations = deque(maxlen=100)
        self.activation_window = timedelta(seconds=30)
        
        # 연관 타입별 기본 강도
        self.connection_strengths = {
            ConnectionType.SEMANTIC: 0.8,
            ConnectionType.TEMPORAL: 0.6,
            ConnectionType.CAUSAL: 0.9,
            ConnectionType.EMOTIONAL: 0.7,
            ConnectionType.SPATIAL: 0.5,
            ConnectionType.PROCEDURAL: 0.4
        }
    
    def add_concept(self, concept: str, metadata: Optional[Dict] = None) -> None:
        """새 개념 추가"""
        if not self.graph.has_node(concept):
            self.graph.add_node(
                concept,
                activation_count=0,
                last_activated=None,
                metadata=metadata or {},
                creation_time=datetime.now()
            )
    
    def activate_concept(self, concept: str, related_concepts: Optional[List[str]] = None) -> None:
        """개념 활성화"""
        current_time = datetime.now()
        
        # 개념 활성화
        if not self.graph.has_node(concept):
            self.add_concept(concept)
        
        # 활성화 정보 업데이트
        self.graph.nodes[concept]['activation_count'] += 1
        self.graph.nodes[concept]['last_activated'] = current_time
        
        # 최근 활성화 기록
        self.recent_activations.append({
            'concept': concept,
            'time': current_time,
            'context': related_concepts or []
        })
        
        # 동시 활성화 분석
        self._analyze_co_activation(concept)
        
        # 확산 활성화 (간접 활성화)
        self._spread_activation(concept)
    
    def connect_concepts(
        self,
        from_concept: str,
        to_concept: str,
        connection_type: ConnectionType = ConnectionType.SEMANTIC,
        custom_strength: Optional[float] = None
    ) -> None:
        """개념 간 연결 생성"""
        # 개념 존재 확인
        self.add_concept(from_concept)
        self.add_concept(to_concept)
        
        # 연결 강도 결정
        strength = custom_strength or self.connection_strengths[connection_type]
        
        if self.graph.has_edge(from_concept, to_concept):
            # 기존 연결 강화
            current_strength = self.graph[from_concept][to_concept]['weight']
            new_strength = min(1.0, current_strength + strength * 0.2)
            self.graph[from_concept][to_concept]['weight'] = new_strength
        else:
            # 새 연결 생성
            self.graph.add_edge(
                from_concept,
                to_concept,
                weight=strength,
                type=connection_type.value,
                creation_time=datetime.now(),
                strengthening_count=0
            )
    
    def find_associations(
        self,
        concepts: List[str],
        depth: int = 2,
        min_strength: Optional[float] = None
    ) -> Dict[str, float]:
        """연관 개념 탐색"""
        min_strength = min_strength or self.min_strength
        related_concepts = {}
        
        for concept in concepts:
            if concept in self.graph:
                # 너비 우선 탐색
                queue = deque([(concept, 0, 1.0)])
                visited = {concept}
                
                while queue:
                    current, current_depth, accumulated_strength = queue.popleft()
                    
                    if current_depth < depth:
                        for neighbor in self.graph.neighbors(current):
                            if neighbor not in visited:
                                edge_strength = self.graph[current][neighbor]['weight']
                                new_strength = accumulated_strength * edge_strength
                                
                                if new_strength >= min_strength:
                                    # 더 강한 경로가 있으면 업데이트
                                    if neighbor not in related_concepts or related_concepts[neighbor] < new_strength:
                                        related_concepts[neighbor] = new_strength
                                    
                                    visited.add(neighbor)
                                    queue.append((neighbor, current_depth + 1, new_strength))
        
        # 자기 자신 제외
        for concept in concepts:
            related_concepts.pop(concept, None)
        
        return related_concepts
    
    def _analyze_co_activation(self, concept: str) -> None:
        """동시 활성화 패턴 분석"""
        current_time = datetime.now()
        co_activated_concepts = []
        
        # 활성화 창 내의 개념들 찾기
        for activation in reversed(self.recent_activations):
            if current_time - activation['time'] <= self.activation_window:
                if activation['concept'] != concept:
                    co_activated_concepts.append(activation['concept'])
                    
                    # 컨텍스트 개념도 추가
                    if activation['context']:
                        co_activated_concepts.extend([
                            c for c in activation['context'] if c != concept
                        ])
            else:
                break
        
        # 동시 활성화된 개념들과 연결 강화
        for co_concept in co_activated_concepts:
            self._strengthen_connection(concept, co_concept, 0.3)
    
    def _spread_activation(self, concept: str, strength: float = 0.1) -> None:
        """확산 활성화 (간접적 활성화)"""
        for neighbor in self.graph.neighbors(concept):
            edge_strength = self.graph[concept][neighbor]['weight']
            indirect_strength = strength * edge_strength
            
            if indirect_strength >= self.min_strength:
                self._strengthen_connection(concept, neighbor, indirect_strength)
    
    def _strengthen_connection(
        self,
        from_concept: str,
        to_concept: str,
        strength_increment: float
    ) -> None:
        """연결 강도 증가"""
        if self.graph.has_edge(from_concept, to_concept):
            current_strength = self.graph[from_concept][to_concept]['weight']
            new_strength = min(1.0, current_strength + strength_increment)
            
            self.graph[from_concept][to_concept]['weight'] = new_strength
            self.graph[from_concept][to_concept]['strengthening_count'] += 1
        else:
            # 연결이 없으면 새로 생성
            self.connect_concepts(from_concept, to_concept, custom_strength=strength_increment)
    
    def apply_decay(self, time_passed: Optional[timedelta] = None) -> None:
        """연결 강도 감소"""
        if time_passed is None:
            time_passed = timedelta(days=1)
        
        decay_factor = self.decay_factor ** (time_passed.days)
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            new_weight = data['weight'] * decay_factor
            
            if new_weight < self.min_strength:
                edges_to_remove.append((u, v))
            else:
                data['weight'] = new_weight
        
        # 약한 연결 제거
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """네트워크 통계"""
        return {
            'total_concepts': self.graph.number_of_nodes(),
            'total_connections': self.graph.number_of_edges(),
            'average_connections': self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
            'most_connected': self._get_most_connected_concepts(5),
            'strongest_connections': self._get_strongest_connections(5)
        }
    
    def _get_most_connected_concepts(self, top_k: int) -> List[Tuple[str, int]]:
        """가장 많이 연결된 개념들"""
        degrees = dict(self.graph.degree())
        return sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def _get_strongest_connections(self, top_k: int) -> List[Tuple[str, str, float]]:
        """가장 강한 연결들"""
        connections = []
        for u, v, data in self.graph.edges(data=True):
            connections.append((u, v, data['weight']))
        
        return sorted(connections, key=lambda x: x[2], reverse=True)[:top_k]