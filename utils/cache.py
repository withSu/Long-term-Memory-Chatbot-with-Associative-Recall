"""
LRU 캐시 구현
자주 접근하는 데이터를 메모리에 캐싱
"""
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """
    Least Recently Used (LRU) 캐시
    제한된 크기로 자주 사용되는 데이터를 캐싱
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """키로 값 조회"""
        if key not in self.cache:
            return None
        
        # 접근 순서 갱신 (최근 사용으로 이동)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """새 항목 저장"""
        if key in self.cache:
            # 기존 항목 갱신
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # 용량 초과 시 가장 오래된 항목 제거
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """캐시 초기화"""
        self.cache.clear()