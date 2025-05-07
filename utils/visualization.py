"""
네트워크 시각화 모듈
연관 네트워크를 시각적으로 표현
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Optional, Dict
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정 개선
def set_korean_font():
    """
    시스템에 설치된 한글 폰트를 찾아 설정
    """
    # 일반 폰트 설정
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 한글 폰트 목록
    korean_fonts = [
        'NanumGothic',
        'Malgun Gothic',
        'Apple SD Gothic Neo',
        'AppleGothic',
        'Noto Sans CJK KR',
        'Gulim',
        'Dotum',
        'Batang'
    ]
    
    # sans-serif에 한글 폰트 추가
    for font_name in korean_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and not font_path.endswith('DejaVuSans.ttf'):
                # sans-serif 패밀리에 한글 폰트 추가
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
                print(f"한글 폰트 설정: {font_name}")
                break
        except Exception as e:
            print(f"폰트 {font_name} 설정 실패: {e}")
            continue
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
    return plt.rcParams['font.sans-serif'][0] if plt.rcParams.get('font.sans-serif') else 'sans-serif'


def visualize_association_network(
    graph: nx.DiGraph,
    center_concept: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[str]:
    """
    연관 네트워크 시각화
    
    Args:
        graph: NetworkX 그래프
        center_concept: 중심 개념
        save_path: 저장 경로
        show: 화면 표시 여부
    
    Returns:
        저장된 파일 경로 또는 None
    """
    # 한글 폰트 설정
    font_name = set_korean_font()

    # 피규어 설정
    plt.figure(figsize=(16, 12), dpi=100)
    
    # 서브그래프 생성 (중심 개념 기준)
    if center_concept and center_concept in graph:
        # 중심에서 2단계 깊이의 노드만 시각화
        subgraph_nodes = set([center_concept])
        subgraph_nodes.update(graph.neighbors(center_concept))
        for node in list(subgraph_nodes):
            subgraph_nodes.update(list(graph.neighbors(node))[:5])  # 최대 5개 이웃
        subgraph = graph.subgraph(subgraph_nodes)
    else:
        subgraph = graph
    
    # 노드 크기 계산 (활성화 횟수 기반)
    node_sizes = []
    for node in subgraph.nodes():
        activation_count = subgraph.nodes[node].get('activation_count', 0)
        size = 500 + min(activation_count * 100, 2000)  # 최대 2500
        node_sizes.append(size)
    
    # 노드 색상 설정
    node_colors = []
    for node in subgraph.nodes():
        if node == center_concept:
            node_colors.append('#ff4444')  # 중심 노드는 빨간색
        else:
            # 접근 최근성에 따른 색상 그라데이션
            last_activated = subgraph.nodes[node].get('last_activated')
            if last_activated:
                time_ago = (datetime.now() - last_activated).total_seconds() / 3600
                fade = min(time_ago / 24, 1.0)  # 24시간 기준으로 페이드
                node_colors.append((0.7 - fade * 0.5, 0.7 - fade * 0.3, 1.0))
            else:
                node_colors.append('#87CEEB')  # 기본 하늘색
    
    # 엣지 두께 설정 (연관 강도 기반)
    edge_widths = []
    edge_colors = []
    for u, v in subgraph.edges():
        weight = subgraph[u][v].get('weight', 0.5)
        edge_widths.append(weight * 8)
        edge_colors.append((0.5, 0.5, 0.5, min(weight * 2, 1.0)))  # 투명도 적용
    
    # 레이아웃 계산
    pos = nx.spring_layout(subgraph, k=1.5/np.sqrt(len(subgraph.nodes())), iterations=50)
    
    # 네트워크 그리기
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_size=node_sizes, 
                          node_color=node_colors,
                          alpha=0.9)
    
    nx.draw_networkx_edges(subgraph, pos, 
                          width=edge_widths,
                          edge_color=edge_colors,
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->')
    
    # 노드 라벨 - font_properties 키워드 제거하고 font_family만 사용
    # 라벨 딕셔너리 직접 생성 (한글 인코딩 처리 위해)
    labels = {}
    for node in subgraph.nodes():
        # 유니코드 문자열로 변환하여 저장
        labels[node] = str(node)
    
    # 라벨 표시 (font_properties 제거)
    nx.draw_networkx_labels(subgraph, pos, labels=labels,
                           font_size=12, 
                           font_weight='bold',
                           font_family=font_name)  # font_properties 제거
    
    # 엣지 가중치 표시
    edge_labels = {}
    for u, v in subgraph.edges():
        weight = subgraph[u][v].get('weight', 0)
        if weight > 0.3:  # 강한 연결만 표시
            edge_labels[(u, v)] = f"{weight:.2f}"
    
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=10, 
                               font_family=font_name)  # 엣지 라벨도 폰트 지정
    
    # 타이틀 설정
    title = f"연관 네트워크 {center_concept} 중심" if center_concept else "연관 네트워크"
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    
    plt.axis('off')
    plt.tight_layout()
    
    # 파일 저장 또는 화면 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path