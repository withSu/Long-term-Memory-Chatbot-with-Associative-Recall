# utils/bloom_filter.py
"""
Bloom 필터 래퍼
빠른 존재 여부 확인을 위한 자료구조
"""
from pybloom_live import BloomFilter as PyBloomFilter
from typing import Any


class BloomFilter:
    """
    Bloom 필터 래퍼 클래스
    set의 멤버십 테스트를 빠르게 수행
    """
    
    def __init__(self, capacity: int = 100000, error_rate: float = 0.001):
        self.filter = PyBloomFilter(capacity=capacity, error_rate=error_rate)
    
    def add(self, item: Any) -> None:
        """항목 추가"""
        self.filter.add(str(item))
    
    def check(self, item: Any) -> bool:
        """항목 존재 여부 확인"""
        return str(item) in self.filter
    
    def clear(self) -> None:
        """필터 초기화"""
        # 새 인스턴스 생성으로 초기화
        capacity = self.filter.capacity
        error_rate = self.filter.error_rate
        self.filter = PyBloomFilter(capacity=capacity, error_rate=error_rate)

