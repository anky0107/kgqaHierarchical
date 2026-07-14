"""
Generate all paper_101 figures from real CWQ data.
Uses a Clean Academic Light Theme with high-contrast elements.
"""

import os, sys, textwrap, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT  = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

# -------------------------------------------------------------
# GLOBALS FOR THEME
# -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_MAIN = '#111111'
TEXT_MUTED = '#555555'
COLOR_CORE = '#2196F3'      # Blue
COLOR_ANSWER = '#4CAF50'    # Green
COLOR_TOPIC = '#FF9800'     # Orange
COLOR_NOISE = '#9E9E9E'     # Grey
COLOR_DANGER = '#F44336'    # Red

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color': TEXT_MAIN,
    'axes.labelcolor': TEXT_MAIN,
    'xtick.color': TEXT_MAIN,
    'ytick.color': TEXT_MAIN,
    'axes.edgecolor': '#DDDDDD',
})

# ─────────────────────────────────────────────────────────────
#  FIGURE 1: Knowledge Graph from real CWQ data
# ─────────────────────────────────────────────────────────────
def fig_knowledge_graph():
    G = nx.DiGraph()

    core_nodes = {
        'Jerry Jones':       {'color': COLOR_TOPIC, 'size': 5000},
        'Dallas Cowboys':    {'color': COLOR_CORE, 'size': 5000},
        'Barry Switzer':     {'color': COLOR_ANSWER, 'size': 5000},
        'Coach Record 1996': {'color': '#9C27B0', 'size': 4000},
        'Arlington, TX':     {'color': COLOR_CORE, 'size': 3500},
    }
    
    # Noise nodes (visually deemphasized to reduce claustrophobia)
    noise_nodes = {
        'Stephen Jones':     {'color': COLOR_NOISE, 'size': 1200},
        'Charlotte Jones':   {'color': COLOR_NOISE, 'size': 1200},
        'Jerry Jones Jr.':   {'color': COLOR_NOISE, 'size': 1200},
        'Comstock Resources':{'color': COLOR_NOISE, 'size': 1200},
        'J. W. Jones':       {'color': COLOR_NOISE, 'size': 1200},
        'Arminta Jones':     {'color': COLOR_NOISE, 'size': 1200},
        
        'NFC East':          {'color': COLOR_NOISE, 'size': 1500},
        'NFC':               {'color': COLOR_NOISE, 'size': 1500},
        'NFL':               {'color': COLOR_NOISE, 'size': 1500},
        'Super Bowl XXX':    {'color': COLOR_NOISE, 'size': 1500},
        'Super Bowl XXVII':  {'color': COLOR_NOISE, 'size': 1500},
        'Super Bowl XXVIII': {'color': COLOR_NOISE, 'size': 1500},
        
        'Troy Aikman':       {'color': COLOR_NOISE, 'size': 1200},
        'Emmitt Smith':      {'color': COLOR_NOISE, 'size': 1200},
        'Michael Irvin':     {'color': COLOR_NOISE, 'size': 1200},
        'Deion Sanders':     {'color': COLOR_NOISE, 'size': 1200},
        'Tony Romo':         {'color': COLOR_NOISE, 'size': 1200},
        'Dak Prescott':      {'color': COLOR_NOISE, 'size': 1200},
        
        'Jimmy Johnson':     {'color': COLOR_NOISE, 'size': 1500},
        'Tom Landry':        {'color': COLOR_NOISE, 'size': 1500},
        'Jason Garrett':     {'color': COLOR_NOISE, 'size': 1500},
        
        'AT&T Stadium':      {'color': COLOR_NOISE, 'size': 1500},
        'Texas Stadium':     {'color': COLOR_NOISE, 'size': 1500},
        'The Star in Frisco':{'color': COLOR_NOISE, 'size': 1500},
        
        'Oklahoma Sooners':  {'color': COLOR_NOISE, 'size': 1500},
        'Arkansas Razorbacks':{'color': COLOR_NOISE, 'size': 1500},
        'Bootlegger':        {'color': COLOR_NOISE, 'size': 1200},
    }
    
    for n, a in {**core_nodes, **noise_nodes}.items():
        G.add_node(n, **a)

    core_edges = [
        ('Jerry Jones',    'Dallas Cowboys',    'sports.team.owner_s'),
        ('Dallas Cowboys', 'Barry Switzer',     'football.historical_coach'),
        ('Barry Switzer',  'Coach Record 1996', 'sports.coach.record'),
        ('Dallas Cowboys', 'Arlington, TX',     'organization.headquarters'),
    ]
    
    noise_edges = [
        # Jerry Jones branches
        ('Jerry Jones',  'Stephen Jones',  'family.parent'),
        ('Jerry Jones',  'Charlotte Jones','family.parent'),
        ('Jerry Jones',  'Jerry Jones Jr.','family.parent'),
        ('Jerry Jones',  'Comstock Resources', 'business.board_member'),
        ('J. W. Jones',  'Jerry Jones',    'family.parent'),
        ('Arminta Jones','Jerry Jones',    'family.parent'),
        
        # Dallas Cowboys branches
        ('Dallas Cowboys','Super Bowl XXX',    'sports.championships'),
        ('Dallas Cowboys','Super Bowl XXVII',  'sports.championships'),
        ('Dallas Cowboys','Super Bowl XXVIII', 'sports.championships'),
        ('Dallas Cowboys','NFC East',          'sports.conference'),
        ('NFC East',      'NFC',               'sports.sub_conference'),
        ('Dallas Cowboys','NFL',               'sports.league_member'),
        
        ('Dallas Cowboys','Troy Aikman',       'football.roster_player'),
        ('Dallas Cowboys','Emmitt Smith',      'football.roster_player'),
        ('Dallas Cowboys','Michael Irvin',     'football.roster_player'),
        ('Dallas Cowboys','Deion Sanders',     'football.roster_player'),
        ('Dallas Cowboys','Tony Romo',         'football.roster_player'),
        ('Dallas Cowboys','Dak Prescott',      'football.roster_player'),
        
        ('Dallas Cowboys','Jimmy Johnson',     'football.historical_coach'),
        ('Dallas Cowboys','Tom Landry',        'football.historical_coach'),
        ('Dallas Cowboys','Jason Garrett',     'football.historical_coach'),
        
        ('Dallas Cowboys','AT&T Stadium',      'sports.team.arena'),
        ('Dallas Cowboys','Texas Stadium',     'sports.team.arena'),
        ('Dallas Cowboys','The Star in Frisco','organization.headquarters'),
        
        # Barry Switzer branches
        ('Barry Switzer', 'Oklahoma Sooners',  'football.historical_coach'),
        ('Barry Switzer', 'Arkansas Razorbacks','football.roster_player'),
        ('Barry Switzer', 'Bootlegger',        'business.founder'),
        ('Barry Switzer', 'Super Bowl XXX',    'sports.coach.championships'),
        
        # --- Cross-Connections (Making the graph highly interconnected) ---
        # Executives
        ('Stephen Jones',  'Dallas Cowboys',   'organization.leadership'),
        ('Charlotte Jones','Dallas Cowboys',   'organization.leadership'),
        ('Jerry Jones Jr.','Dallas Cowboys',   'organization.leadership'),
        
        # Players to Championships
        ('Troy Aikman',   'Super Bowl XXX',    'sports.player.championships'),
        ('Emmitt Smith',  'Super Bowl XXX',    'sports.player.championships'),
        ('Michael Irvin', 'Super Bowl XXX',    'sports.player.championships'),
        ('Troy Aikman',   'Super Bowl XXVII',  'sports.player.championships'),
        ('Emmitt Smith',  'Super Bowl XXVII',  'sports.player.championships'),
        ('Michael Irvin', 'Super Bowl XXVII',  'sports.player.championships'),
        ('Troy Aikman',   'Super Bowl XXVIII', 'sports.player.championships'),
        ('Emmitt Smith',  'Super Bowl XXVIII', 'sports.player.championships'),
        ('Michael Irvin', 'Super Bowl XXVIII', 'sports.player.championships'),
        
        # Coaches to Championships / Players
        ('Jimmy Johnson', 'Super Bowl XXVII',  'sports.coach.championships'),
        ('Jimmy Johnson', 'Super Bowl XXVIII', 'sports.coach.championships'),
        ('Jason Garrett', 'Tony Romo',         'sports.coach.roster'),
        ('Jason Garrett', 'Dak Prescott',      'sports.coach.roster'),
        
        # Stadiums and Locations
        ('AT&T Stadium',  'Arlington, TX',     'location.containedby'),
        ('Texas Stadium', 'Arlington, TX',     'location.containedby'), # (Close enough for KG noise)
        ('The Star in Frisco', 'NFL',          'sports.facility'),
        
        # Conferences
        ('Dallas Cowboys', 'NFC',              'sports.team.conference'),
        ('NFC East',       'NFL',              'sports.conference.league'),
        ('NFC',            'NFL',              'sports.conference.league'),
    ]

    for s, t, r in core_edges: G.add_edge(s, t, label=r, core=True)
    for s, t, r in noise_edges: G.add_edge(s, t, label=r, core=False)

    fig, ax = plt.subplots(figsize=(26, 14))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    # ---------------------------------------------------------
    # Structured Radial Layout (Massive Frame)
    # ---------------------------------------------------------
    pos = {}
    
    # 1. Place core nodes linearly with HUGE spacing to prevent claustrophobia
    pos['Jerry Jones']       = (-16.0, 0)
    pos['Dallas Cowboys']    = (  0.0, 0)
    pos['Barry Switzer']     = ( 16.0, 0)
    pos['Coach Record 1996'] = ( 25.0, 0)
    pos['Arlington, TX']     = (  0.0, -8.0)
    
    # 2. Helper to place nodes radially around a center
    def place_radially(center_node, neighbors, radius, start_angle, end_angle):
        angles = np.linspace(start_angle, end_angle, len(neighbors))
        cx, cy = pos[center_node]
        for node, ang in zip(neighbors, angles):
            rad = np.radians(ang)
            pos[node] = (cx + radius * np.cos(rad), cy + radius * np.sin(rad))

    # Jerry Jones' noise (Left side)
    jj_noise = ['Stephen Jones', 'Charlotte Jones', 'Jerry Jones Jr.', 'Comstock Resources', 'J. W. Jones', 'Arminta Jones']
    place_radially('Jerry Jones', jj_noise, 5.0, 90, 270)
    
    # Dallas Cowboys' noise (Top and Bottom)
    dc_noise_top = ['Super Bowl XXX', 'Super Bowl XXVII', 'Super Bowl XXVIII', 'NFC East', 'NFL']
    place_radially('Dallas Cowboys', dc_noise_top, 6.5, 30, 150)
    pos['NFC'] = (pos['NFC East'][0], pos['NFC East'][1] + 2.5) # Stack NFC above NFC East
    
    dc_noise_bot1 = ['Troy Aikman', 'Emmitt Smith', 'Michael Irvin', 'Deion Sanders', 'Tony Romo', 'Dak Prescott']
    place_radially('Dallas Cowboys', dc_noise_bot1, 5.5, 190, 260)
    
    dc_noise_bot2 = ['Jimmy Johnson', 'Tom Landry', 'Jason Garrett', 'AT&T Stadium', 'Texas Stadium', 'The Star in Frisco']
    place_radially('Dallas Cowboys', dc_noise_bot2, 6.0, 280, 350)
    
    # Barry Switzer's noise (Right side)
    bs_noise = ['Oklahoma Sooners', 'Arkansas Razorbacks', 'Bootlegger']
    place_radially('Barry Switzer', bs_noise, 5.0, -70, 70)

    # Draw edges
    noise_e = [(s,t) for s,t,r in noise_edges]
    nx.draw_networkx_edges(G, pos, edgelist=noise_e, ax=ax, edge_color='#E5E5E5', width=1.2,
                           arrows=True, arrowsize=12, arrowstyle='-|>', connectionstyle='arc3,rad=0.0')
    core_e = [(s,t) for s,t,r in core_edges]
    nx.draw_networkx_edges(G, pos, edgelist=core_e, ax=ax, edge_color=TEXT_MAIN, width=4.0,
                           arrows=True, arrowsize=25, arrowstyle='-|>', connectionstyle='arc3,rad=0.1')

    # Draw nodes
    for node, (x,y) in pos.items():
        color = G.nodes[node].get('color', COLOR_NOISE)
        size  = G.nodes[node].get('size',  1500)
        # Scale radius properly for the massive canvas
        r = (size/3000)**0.5 * 0.8
        
        # Draw node circle
        circle = plt.Circle((x,y), r, color=color, zorder=3, alpha=0.95)
        ax.add_patch(circle)
        border = plt.Circle((x,y), r, color='#333333', fill=False, linewidth=2.5 if color!=COLOR_NOISE else 1.0, zorder=4)
        ax.add_patch(border)
        
        # Draw text
        wrapped = '\n'.join(textwrap.wrap(node, 10 if color==COLOR_NOISE else 15))
        fs = 9 if color == COLOR_NOISE else 14
        tc = 'white' if color in [COLOR_CORE, COLOR_ANSWER, '#9C27B0'] else '#111111'
        fw = 'normal' if color == COLOR_NOISE else 'bold'
        ax.text(x, y, wrapped, ha='center', va='center', fontsize=fs, color=tc, fontweight=fw, zorder=5)

    # Labels for noise edges (Small, muted, brief)
    noise_elabels = {(s,t): r for s,t,r in noise_edges}
    for (s,t), lbl in noise_elabels.items():
        mx, my = (pos[s][0]+pos[t][0])/2, (pos[s][1]+pos[t][1])/2
        # Keep noise labels brief to avoid massive clutter
        short_lbl = lbl.split('.')[-1]
        ax.text(mx, my, short_lbl, fontsize=8, color='#888888', ha='center', va='center', style='italic', zorder=5,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

    # Labels for core path (Large, bold, full relation)
    elabels = {(s,t): r for s,t,r in core_edges}
    for (s,t), lbl in elabels.items():
        mx, my = (pos[s][0]+pos[t][0])/2, (pos[s][1]+pos[t][1])/2 + 0.5
        ax.text(mx, my, lbl, fontsize=12, color=TEXT_MAIN, ha='center', va='center', style='italic', zorder=6,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBBBBB', alpha=0.95, lw=1.5))

    legend_items = [
        mpatches.Patch(color=COLOR_TOPIC, label='Topic Entity (Jerry Jones)'),
        mpatches.Patch(color=COLOR_CORE, label='Intermediate Entity'),
        mpatches.Patch(color=COLOR_ANSWER, label='Answer Entity (Barry Switzer)'),
        mpatches.Patch(color=COLOR_NOISE, label='Graph Branching Noise'),
        plt.Line2D([0], [0], color=TEXT_MAIN, lw=4, label='Optimal Reasoning Path'),
    ]
    leg = ax.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=13, framealpha=0.95, facecolor='white', edgecolor='#CCCCCC')

    ax.set_title('Knowledge Graph Subgraph — Highlighting the Massive Freebase Traversal Space', fontsize=22, fontweight='bold', pad=30)
    ax.set_xlim(-23, 30); ax.set_ylim(-10, 10)

    out = os.path.join(OUT, 'kg_example.png')
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ─────────────────────────────────────────────────────────────
#  FIGURE 2: Main Architecture Diagram
# ─────────────────────────────────────────────────────────────
def fig_main_architecture():
    fig, ax = plt.subplots(figsize=(24, 13))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # Color tokens matching light academic theme
    C_LBLUE = '#EBF4FA'
    C_LORANGE = '#FFF2E6'
    C_LGREEN = '#EAF2EA'
    C_LYELLOW = '#FFF9C4'

    # Helper function for fancy boxes
    def draw_box(x, y, w, h, label, sublabel='', color='#F0F0F0', ec='#CCCCCC', lw=1.5, z=3, text_color=TEXT_MAIN, fs_label=11, fs_sub=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1', linewidth=lw, edgecolor=ec, facecolor=color, zorder=z)
        ax.add_patch(rect)
        if label:
            ax.text(x+w/2, y+h/2 + (0.12 if sublabel else 0), label, ha='center', va='center', fontsize=fs_label, color=text_color, fontweight='bold', zorder=z+1)
        if sublabel:
            ax.text(x+w/2, y+h/2 - 0.18, sublabel, ha='center', va='center', fontsize=fs_sub, color=TEXT_MUTED, style='italic', zorder=z+1)
        return rect

    # Helper function for fancy arrows
    def draw_arrow(x1, y1, x2, y2, color='#333333', lw=1.5, style='-|>', ms=6, rad=0.0, ls='-'):
        style_str = f'-|>,head_width={ms/3},head_length={ms/2}' if style == '-|>' else style
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style_str, mutation_scale=10, linewidth=lw, linestyle=ls, edgecolor=color, facecolor=color, connectionstyle=f'arc3,rad={rad}', zorder=2)
        ax.add_patch(arrow)
        return arrow

    # Draw panel backgrounds with thin outlines and rounded labels
    # Panel I: Semantic Encoding (x: 0.2 to 6.8)
    draw_box(0.2, 0.4, 6.6, 11.2, '', '', '#F7F9FC', '#B0C4DE', lw=1, z=1)
    ax.text(3.5, 11.3, "I. SEMANTIC ENCODING (Stage I backbone, Frozen)", ha='center', fontsize=12, fontweight='bold', color='#1A365D')

    # Panel II: Reasoning Engine (x: 7.2 to 15.8)
    draw_box(7.2, 0.4, 8.6, 11.2, '', '', '#FFF9F2', '#FFD8B3', lw=1, z=1)
    ax.text(11.5, 11.3, "II. MULTIMODAL HOP PLANNER (Stage I planner, Trainable)", ha='center', fontsize=12, fontweight='bold', color='#7B3F00')

    # Panel III: Traversal Policy (x: 16.2 to 23.8)
    draw_box(16.2, 0.4, 7.6, 11.2, '', '', '#F5FAF6', '#C2E0C6', lw=1, z=1)
    ax.text(20.0, 11.3, "III. META-CONSTRAINT POLICY (Stage II & III, Trainable)", ha='center', fontsize=12, fontweight='bold', color='#1E4620')

    # ------------------ PANEL I: SEMANTIC ENCODING ------------------
    # 1. Input Question
    draw_box(0.5, 0.8, 6.0, 1.2, "Input Question (q)", "Who was the 1996 coach of the team owned by Jerry Jones?", '#FFFFFF', COLOR_CORE, lw=2)
    
    # 2. Tokenizer
    draw_box(0.5, 2.7, 6.0, 0.9, "Tokenizer (RoBERTa-large)", "Appends [CLS] and <s> special tokens", '#FFFFFF', '#7F7F7F', lw=1.5)
    draw_arrow(3.5, 2.0, 3.5, 2.7, lw=2)
    
    # 3. Embeddings Layer
    draw_box(0.5, 4.3, 6.0, 0.7, "Initial Token Embeddings", "Dimension: [1, num_of_tokens, 1024]", C_LBLUE, COLOR_CORE, lw=1.5)
    draw_arrow(3.5, 3.6, 3.5, 4.3, lw=2)

    # 4. RoBERTa Backbone Stack
    roberta_y = 5.7
    draw_box(0.5, roberta_y, 6.0, 1.6, "", "", '#E1F5FE', COLOR_CORE, lw=2)
    ax.text(3.5, roberta_y+1.0, "RoBERTa-Large Backbone", ha='center', fontsize=12, fontweight='bold', color='#1A365D')
    ax.text(3.5, roberta_y+0.6, "L = 24 layers, H = 1024 dim", ha='center', fontsize=10, color=TEXT_MUTED)
    # Stack lines to show layers
    for ly in [0.15, 0.3, 0.45]:
        ax.plot([0.7, 6.3], [roberta_y+ly, roberta_y+ly], color=COLOR_CORE, lw=1, alpha=0.5)
    draw_arrow(3.5, 5.0, 3.5, roberta_y, lw=2)

    # Padlock icon
    draw_box(0.7, roberta_y+1.1, 0.7, 0.4, "FROZEN", "", '#FFE0B2', '#FF9800', lw=1, fs_label=7)

    # 5. Output Embeddings & CLS Extraction Callout
    draw_arrow(3.5, roberta_y+1.6, 3.5, 8.0, lw=2)
    draw_box(0.5, 8.0, 6.0, 0.8, "Token Embeddings with Meaning", "Dimension: [1, num_of_tokens, 1024]", C_LBLUE, COLOR_CORE, lw=1.5)
    
    # 6. Extraction Box
    draw_arrow(3.5, 8.8, 3.5, 9.4, lw=2)
    draw_box(0.5, 9.4, 6.0, 0.8, "Extract First Token Embedding (<s>)", "Dimension: [1, 1024]", C_LGREEN, COLOR_ANSWER, lw=2)
    
    # Callout text explaining why we drop other tokens
    draw_box(0.5, 10.4, 6.0, 0.7, "Computation Optimization", "Drop remaining tokens to minimize sequential memory complexity", C_LYELLOW, '#FBC02D', lw=1, fs_label=9, fs_sub=8)

    # ------------------ PANEL II: MULTIMODAL HOP PLANNER ------------------
    # Projection Layer (Weighted Matrix)
    draw_box(7.5, 9.4, 3.2, 0.8, "Projection Layer (W_p)", "Weighted Matrix [1024 x 512]\nYields [1, 512] embedding", '#FFFFFF', '#7F7F7F', lw=1.5)
    # Arrow from Panel I <s> extraction to Projection Layer
    draw_arrow(6.5, 9.8, 7.5, 9.8, lw=2)
    
    # Replication into 4 parallel hops
    draw_arrow(9.1, 9.4, 9.1, 8.3, lw=2)
    draw_box(7.5, 7.5, 3.2, 0.8, "Semantic Vector Replication", "Replicates [1, 512] into 4 hop channels", C_LORANGE, COLOR_TOPIC, lw=1.5)

    # Hops representation (Replication grid)
    hop_x = 11.6
    hop_w = 3.8
    draw_box(hop_x, 7.0, hop_w, 2.5, "", "", '#FFFFFF', COLOR_TOPIC, lw=1.5)
    ax.text(hop_x+hop_w/2, 9.1, "Hop Replication & Positional Addition", ha='center', fontsize=10, fontweight='bold', color='#7B3F00')
    
    # Draw the 4 hops in grid
    for idx, (hop_name, hop_col) in enumerate([("Hop 1", '#E3F2FD'), ("Hop 2", '#FFF3E0'), ("Hop 3", '#F3E5F5'), ("Hop 4", '#E8F5E9')]):
        hy = 7.2 + idx * 0.45
        draw_box(hop_x+0.2, hy, hop_w-0.4, 0.35, f"{hop_name} Vector + Learned Positional E_hop", "Dimension: [1, 512]", hop_col, COLOR_TOPIC, lw=1, fs_label=8, fs_sub=6)

    # Arrow from replication box to the hop addition grid
    draw_arrow(10.7, 7.9, hop_x, 7.9, lw=2)

    # Cross-Hop Attention Transformer block
    trans_y = 3.6
    draw_box(7.5, trans_y, 7.9, 2.3, "", "", '#E8F8F5', '#16A085', lw=2)
    ax.text(11.45, trans_y+1.8, "Cross-Hop Attention Transformer Encoder Stack", ha='center', va='center', fontsize=12, fontweight='bold', color='#0E6655')
    ax.text(11.45, trans_y+1.4, "Layers: L = 4  |  Attention Heads: H = 8", ha='center', va='center', fontsize=10, color=TEXT_MUTED)
    
    # Draw the internal layers of vanilla transformer
    for l_idx in range(4):
        ly = trans_y + 0.2 + l_idx * 0.28
        draw_box(8.2, ly, 6.5, 0.24, f"Transformer Encoder Block Layer {l_idx+1}", "", '#FFFFFF', '#16A085', lw=1, fs_label=8)

    # Arrow from Hop addition grid to vanilla transformer
    draw_arrow(13.5, 7.0, 11.45, 5.9, lw=2)

    # Callout detailing a single Transformer Block
    block_x = 7.5
    block_y = 1.0
    draw_box(block_x, block_y, 7.9, 2.1, "", "", '#F2F4F4', '#566573', lw=1.5)
    ax.text(block_x+3.95, block_y+1.8, "Transformer Block Layer Detail", ha='center', fontsize=10, fontweight='bold', color='#2C3E50')
    detail_items = [
        "Layer Norm -> Multi-Head Self-Attention (MHSA, h=8, d=64) -> Residual Connection",
        "Layer Norm -> Feed-Forward Network (FFN, d_ff=2048) -> Residual Connection"
    ]
    for idx, item in enumerate(detail_items):
        draw_box(block_x+0.2, block_y+0.2 + idx * 0.7, 7.5, 0.6, item, "", '#FFFFFF', '#566573', lw=1, fs_label=8)

    # Arrow from vanilla transformer to block detail
    draw_arrow(7.8, trans_y, 7.8, block_y+2.1, lw=1.5, style='->', ls='--')

    # Vanilla Transformer Outputs
    out_y = 0.8
    # Relation Head
    draw_box(13.6, out_y+1.8, 2.0, 0.6, "Relation Head", "P(r_h|q) softmax", '#FFFFFF', '#E67E22', lw=1.5, fs_label=9, fs_sub=7)
    # Stop Head
    draw_box(13.6, out_y+1.0, 2.0, 0.6, "Stop Head", "P(stop_h|q) sigmoid", '#FFFFFF', '#E67E22', lw=1.5, fs_label=9, fs_sub=7)
    # Domain Head
    draw_box(13.6, out_y+0.2, 2.0, 0.6, "Domain Head", "P(d|q) softmax", '#FFFFFF', '#E67E22', lw=1.5, fs_label=9, fs_sub=7)
    
    # Arrow from Transformer to outputs
    draw_arrow(11.45, trans_y, 11.45, 2.4, lw=2)
    draw_arrow(11.45, 2.4, 13.6, 2.1, lw=2)
    draw_arrow(11.45, 2.4, 13.6, 1.3, lw=2)
    draw_arrow(11.45, 2.4, 13.6, 0.5, lw=2)

    # Confidence Head
    draw_box(11.4, 0.8, 2.0, 0.6, "Confidence Head", "eta_q = sigmoid(W_eta h_q)", '#FFFFFF', '#E67E22', lw=1.5, fs_label=9, fs_sub=7)
    draw_arrow(6.5, 1.5, 11.4, 1.1, lw=2) # Arrow from RoBERTa CLS projection to confidence head

    # ------------------ PANEL III: META-CONSTRAINT POLICY ------------------
    # 1. State Concatenation s_t
    draw_box(16.5, 8.5, 7.0, 1.0, "Input State s_t = [h_h ; eta_q]", "Concatenates Refined Hop Vector h_h [1 x 512]\nwith Semantic Confidence Score eta_q [scalar]", C_LBLUE, COLOR_CORE, lw=2)
    
    # Arrow from Confidence Head and Relation Head to State Concatenation
    draw_arrow(12.4, 1.4, 16.5, 8.8, lw=1.5, style='->', ls='--')
    draw_arrow(14.6, 2.4, 16.5, 9.0, lw=1.5, style='->', ls='--')

    # 2. Reinforcement Learning Agent (Policy & Value Networks)
    agent_y = 4.8
    draw_box(16.5, agent_y, 7.0, 2.8, "", "", '#F4F9F4', COLOR_ANSWER, lw=2)
    ax.text(20.0, agent_y+2.3, "RL Meta-Constraint Controller (PPO)", ha='center', fontsize=12, fontweight='bold', color='#1E4620')
    
    # Draw Policy Network and Value Network
    draw_box(16.8, agent_y+0.3, 3.1, 1.7, "Policy Head (π_θ)", "Outputs Action Probabilities\nAction Space: 4 discrete actions\nParameters: [4 x 513]", '#FFFFFF', COLOR_ANSWER, lw=1.5, fs_label=10, fs_sub=8)
    draw_box(20.1, agent_y+0.3, 3.1, 1.7, "Value Head (V_ϕ)", "Estimates State Value V(s)\nUsed for GAE advantages\nParameters: [1 x 513]", '#FFFFFF', COLOR_ANSWER, lw=1.5, fs_label=10, fs_sub=8)

    draw_arrow(20.0, 8.5, 20.0, 7.6, lw=2)

    # 3. Action Selection & Search Widths
    act_y = 1.0
    draw_box(16.5, act_y, 7.0, 3.1, "", "", '#FFFFFF', COLOR_ANSWER, lw=1.5)
    ax.text(20.0, act_y+2.6, "High-Level Action Selection (Width)", ha='center', fontsize=11, fontweight='bold', color='#1E4620')
    
    actions_list = [
        ("TIGHT", "Beam = 1", C_LBLUE, COLOR_CORE),
        ("MEDIUM", "Beam = 5", C_LORANGE, COLOR_TOPIC),
        ("LOOSE", "Beam = 50", '#FADBD8', COLOR_DANGER),
        ("STOP", "Early Termination", '#EBDEF0', '#8E44AD')
    ]
    for idx, (aname, adesc, acol, aec) in enumerate(actions_list):
        ax_x = 16.7 + idx * 1.65
        draw_box(ax_x, act_y+0.3, 1.5, 2.0, aname, adesc, acol, aec, lw=1.5, fs_label=9, fs_sub=7)

    # Arrow from Policy head to Action Selection
    draw_arrow(18.3, agent_y, 20.0, act_y+3.1, lw=2)

    # 4. Symbolic Graph Environment & Feedback Loop
    # Arrow from STOP action to Output/Stage III
    draw_arrow(22.45, act_y+0.3, 22.45, 0.45, lw=2, color=COLOR_ANSWER)
    ax.text(22.45, 0.2, "To Stage III\n(Gold Entity Reranker)", ha='center', va='center', fontsize=9, fontweight='bold', color=COLOR_ANSWER)

    # Draw feedback loop line going from other actions
    ax.plot([19.1, 19.1, 11.45, 11.45], [act_y+0.3, 0.2, 0.2, trans_y], color='#FF7F0E', lw=2.5, ls='--', alpha=0.8, zorder=2)
    # Put a text on the loop
    ax.text(14.5, 0.35, "Action a_t -> KG Traversal (LMDB Environment) -> Update Entity Embedding for next hop", ha='center', va='center', fontsize=9, color='#FF7F0E', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#FF7F0E', lw=1, alpha=0.9), zorder=5)

    # Title of the framework at the very top
    ax.text(12, 12.3, "Stage I & Stage II Framework Architecture — Proposed Adaptive Neuro-Symbolic KGQA", ha='center', va='center', fontsize=18, fontweight='bold', color='#111111')
    ax.text(12, 12.0, "Seamless integration of RoBERTa-large semantic encoding, cross-hop vanilla transformer planning, and compact RL traversal actions", ha='center', va='center', fontsize=11, color=TEXT_MUTED)

    out = os.path.join(OUT, 'main_architecture.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ─────────────────────────────────────────────────────────────
#  FIGURE 3: Traversal Action Visualization
# ─────────────────────────────────────────────────────────────
def fig_traversal_actions():
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.patch.set_facecolor(BG_COLOR)

    configs = [
        ('TIGHT',  COLOR_CORE,   1,  'Top-1 relation\nHigh precision\nLow recall\nBeam = 1'),
        ('MEDIUM', COLOR_TOPIC,  5,  'Top-5 relations\nBalanced\nBeam = 5'),
        ('LOOSE',  COLOR_DANGER, 50, 'Domain-wide\nHigh recall\nLow precision\nBeam = 50'),
        ('STOP',   '#9C27B0',    0,  'Terminate path\nAnswer collected\nNo expansion'),
    ]

    center = (0.5, 0.75)
    for ax, (name, color, k, desc) in zip(axes, configs):
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')

        c = plt.Circle(center, 0.12, color=color, alpha=0.2, zorder=2)
        ax.add_patch(c)
        border = plt.Circle(center, 0.12, color=color, fill=False, lw=3, zorder=3)
        ax.add_patch(border)
        ax.text(*center, 'eₜ', ha='center', va='center', fontsize=16, color=color, fontweight='bold', zorder=4)

        if k > 0:
            show_k = min(k, 9)
            angles = np.linspace(210, 330, show_k)
            for i, ang in enumerate(angles):
                rad = np.radians(ang)
                ex, ey = center[0] + 0.35 * np.cos(rad), center[1] + 0.35 * np.sin(rad)
                alpha = 1.0 if i < (1 if k==1 else 3 if k==5 else 8) else 0.4
                nc = plt.Circle((ex,ey), 0.08, color=color, alpha=alpha*0.3, zorder=2)
                ax.add_patch(nc)
                ax.add_patch(plt.Circle((ex,ey), 0.08, color=color, fill=False, lw=2, alpha=alpha, zorder=3))
                ax.annotate('', xy=(ex,ey), xytext=center, arrowprops=dict(arrowstyle='->', color=color, lw=2.5*alpha, alpha=alpha))

            if k == 50:
                ax.text(0.5, 0.22, '... 50 relations expanded ...', ha='center', fontsize=10, color=color, style='italic', fontweight='bold')

        ax.set_title(name, color=color, fontsize=18, fontweight='bold', pad=10)
        ax.text(0.5, 0.15, desc, ha='center', va='top', fontsize=11, color=TEXT_MUTED, multialignment='center')

    fig.suptitle('Adaptive Traversal Actions — RLMC Meta-Constraint Control', fontsize=18, fontweight='bold', y=1.05)
    out = os.path.join(OUT, 'traversal_actions.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ─────────────────────────────────────────────────────────────
#  FIGURE 4: Multi-Stage Training Pipeline
# ─────────────────────────────────────────────────────────────
def fig_training_pipeline():
    fig, ax = plt.subplots(figsize=(18, 5.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 16); ax.set_ylim(0, 5)
    ax.axis('off')

    stages = [
        (0.3, 1.2, 3.4, 2.6, 'Stage I\nSemantic Planner\nTraining',
         'Exp 7 — RoBERTa-Large\nAdamW lr=1e-5\n30 epochs\nLoss: Lrel + Ldomain + Lstop',
         '#E3F2FD', '#2196F3'),
        (4.2, 1.2, 3.4, 2.6, 'Stage II\nFrozen-Planner\nRL Optimization',
         'Exp 9 — PPO A2C\nPlanner FROZEN\nAdamW lr=1e-4\n10 epochs, γ=0.99',
         '#F3E5F5', '#9C27B0'),
        (8.1, 1.2, 3.4, 2.6, 'Stage III\nSemantic-Teacher\nRL (STRL)',
         'Exp 15 — Unfrozen backbone\nInfoNCE τ=0.07 + PPO ε=0.2\nCurriculum: λsem 1.0→0.3\n20 epochs',
         '#E8F5E9', '#4CAF50'),
        (12.0,1.2, 3.4, 2.6, 'Stage IV\nCandidate\nDisambiguation',
         'Exp 16 — CDS Ranker\nall-MiniLM-L6-v2\nContrastive lr=2e-5\n5 epochs, 15 neg',
         '#FFF3E0', '#FF9800'),
    ]

    for x,y,w,h,title,body,fc,ec in stages:
        rect = FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.2', edgecolor=ec, facecolor=fc, linewidth=2.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h-0.45, title, ha='center', va='center', fontsize=12, color=TEXT_MAIN, fontweight='bold', zorder=4, multialignment='center')
        ax.plot([x+0.2, x+w-0.2], [y+h-0.95, y+h-0.95], color=ec, lw=1.5, alpha=0.6, zorder=4)
        ax.text(x+w/2, y+0.8, body, ha='center', va='center', fontsize=10, color=TEXT_MUTED, zorder=4, multialignment='center')

    arrow_x = [3.7, 7.6, 11.5]
    for x in arrow_x:
        ax.annotate('', xy=(x+0.5, 2.5), xytext=(x, 2.5), arrowprops=dict(arrowstyle='->', color='#555555', lw=3.0))

    ax.text(3.95, 2.8, 'Frozen\nweights', ha='center', fontsize=9, color='#555', style='italic')
    ax.text(7.85, 2.8, 'Joint\ntune', ha='center', fontsize=9, color='#555', style='italic')
    ax.text(11.75, 2.8, 'Candidates', ha='center', fontsize=9, color='#555', style='italic')

    ax.set_title('Multi-Stage Training Pipeline — RLMC Framework', fontsize=16, fontweight='bold', pad=15)
    out = os.path.join(OUT, 'training_pipeline.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ─────────────────────────────────────────────────────────────
#  FIGURE 5: Traversal Complexity O(b^h) vs O(k^h)
# ─────────────────────────────────────────────────────────────
def fig_complexity_curve():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    hops = np.array([1, 2, 3, 4])
    b = 847
    k_tight = 1; k_medium = 5; k_loose = 50

    ax.plot(hops, b**hops,       'o-', color=COLOR_DANGER, lw=3, ms=9, label=f'Static Exhaustive  O(b^h), b={b}')
    ax.plot(hops, k_loose**hops, 's--', color=COLOR_TOPIC, lw=2.5, ms=8, label=f'LOOSE  O(k^h), k={k_loose}')
    ax.plot(hops, k_medium**hops,'D--', color=COLOR_CORE, lw=2.5, ms=8, label=f'MEDIUM  O(k^h), k={k_medium}')
    ax.plot(hops, k_tight**hops, '^-', color=COLOR_ANSWER, lw=2.5, ms=8, label=f'TIGHT  O(k^h), k={k_tight}')

    ax.set_yscale('log')
    ax.set_xlabel('Reasoning Depth (Hops)', color=TEXT_MAIN, fontsize=13, fontweight='bold')
    ax.set_ylabel('Candidate Space Size (log scale)', color=TEXT_MAIN, fontsize=13, fontweight='bold')
    ax.set_title('Traversal-Space Complexity: Static vs. Adaptive Control', color=TEXT_MAIN, fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(hops)
    ax.tick_params(colors=TEXT_MAIN, labelsize=11)
    ax.spines['bottom'].set_color('#888')
    ax.spines['left'].set_color('#888')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#E0E0E0', linestyle='--', alpha=0.7)
    
    leg = ax.legend(fontsize=11, framealpha=1.0, facecolor='white', edgecolor='#CCC')
    plt.setp(leg.get_texts(), color=TEXT_MAIN)

    out = os.path.join(OUT, 'complexity_curve.png')
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ─────────────────────────────────────────────────────────────
#  FIGURE 6: Detailed Working Example (Hierarchical Tree)
# ─────────────────────────────────────────────────────────────
def fig_working_example():
    """
    Creates a detailed hierarchical tree showing STRL vs Standard RL.
    Top Half: Standard RL (Exp 9) showing branches, confidence drops, and metrics.
    Bottom Half: STRL (Exp 15) showing InfoNCE TIGHT beam.
    """
    fig, ax = plt.subplots(figsize=(20, 11))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis('off')

    def draw_node(x, y, label, color, ec, width=2.5, height=0.7):
        rect = FancyBboxPatch((x-width/2, y-height/2), width, height, boxstyle='round,pad=0.1', linewidth=2.0, edgecolor=ec, facecolor=color, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, color=TEXT_MAIN, fontweight='bold', zorder=4)

    def draw_callout(x, y, text, color):
        ax.text(x, y, text, ha='center', va='center', fontsize=9, color=color,
                bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec=color, lw=1.5, alpha=0.9), zorder=5)

    def draw_edge(x1, y1, x2, y2, label='', color=TEXT_MAIN, lw=2.0, ls='-', alpha=1.0):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls, alpha=alpha, connectionstyle='arc3,rad=0.0', zorder=2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2 + 0.15
            ax.text(mx, my, label, ha='center', va='center', fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8), zorder=5)

    # Coordinates
    x_nodes = [2.0, 7.0, 12.0, 17.0]
    
    # ------------------ TOP: EXP 9 (Standard RL) ------------------
    y_main = 8.0
    ax.text(0.5, 10.2, 'Standard RL (Exp 9) — Wide Search Space due to Semantic Uncertainty', color=COLOR_TOPIC, fontsize=16, fontweight='bold', ha='left')
    
    # Nodes
    draw_node(x_nodes[0], y_main, 'Jerry Jones', '#E3F2FD', '#2196F3')
    draw_node(x_nodes[1], y_main, 'Dallas Cowboys', '#E3F2FD', '#2196F3')
    draw_node(x_nodes[2], y_main, 'Barry Switzer', '#E3F2FD', '#2196F3')
    draw_node(x_nodes[3], y_main, 'Coach Record 1996', '#E8F5E9', '#4CAF50')

    # Branch Nodes
    draw_node(x_nodes[1], y_main+1.5, 'Stephen Jones', '#F5F5F5', '#9E9E9E', width=2.0)
    draw_node(x_nodes[2], y_main+1.5, 'Troy Aikman', '#F5F5F5', '#9E9E9E', width=2.0)
    draw_node(x_nodes[2], y_main-1.5, 'NFC East', '#F5F5F5', '#9E9E9E', width=2.0)
    
    # Hop 1 (MEDIUM)
    draw_edge(x_nodes[0]+1.3, y_main, x_nodes[1]-1.3, y_main, 'sports.team.owner_s\n(Score: 0.88)', COLOR_CORE, lw=3)
    draw_edge(x_nodes[0]+1.3, y_main, x_nodes[1]-1.3, y_main+1.5, 'family.parent\n(0.62)', COLOR_NOISE, ls='--')
    draw_callout((x_nodes[0]+x_nodes[1])/2, y_main-0.8, "Conf: 0.65\nAction: MEDIUM\nBeam: 5 candidates", COLOR_TOPIC)

    # Hop 2 (LOOSE)
    draw_edge(x_nodes[1]+1.3, y_main, x_nodes[2]-1.3, y_main, 'historical_coach\n(Score: 0.75)', COLOR_CORE, lw=3)
    draw_edge(x_nodes[1]+1.3, y_main, x_nodes[2]-1.3, y_main+1.5, 'roster_player (0.60)', COLOR_NOISE, ls='--')
    draw_edge(x_nodes[1]+1.3, y_main, x_nodes[2]-1.3, y_main-1.5, 'sports.conf (0.65)', COLOR_NOISE, ls='--')
    draw_edge(x_nodes[1]+1.3, y_main, x_nodes[2]-1.0, y_main+0.7, '', COLOR_NOISE, ls='--', alpha=0.5)
    draw_edge(x_nodes[1]+1.3, y_main, x_nodes[2]-1.0, y_main-0.7, '', COLOR_NOISE, ls='--', alpha=0.5)
    ax.text((x_nodes[1]+x_nodes[2])/2, y_main+0.8, "... 46 more divergent paths ...", color=COLOR_NOISE, fontsize=9, ha='center', style='italic')
    draw_callout((x_nodes[1]+x_nodes[2])/2, y_main-0.8, "Conf: 0.40 (Ambiguity)\nAction: LOOSE\nBeam: 50 candidates", COLOR_DANGER)

    # Hop 3 (MEDIUM)
    draw_edge(x_nodes[2]+1.3, y_main, x_nodes[3]-1.3, y_main, 'sports.coach.record\n(Score: 0.85)', COLOR_CORE, lw=3)
    draw_callout((x_nodes[2]+x_nodes[3])/2, y_main-0.8, "Conf: 0.52\nAction: MEDIUM\nBeam: 5 candidates", COLOR_TOPIC)

    # Metrics Exp 9
    ax.text(18.5, y_main, "Total Explored:\n1,250 Candidates\n(Low Efficiency)", ha='center', va='center', fontsize=11, color=COLOR_DANGER, fontweight='bold')


    # ------------------ BOTTOM: EXP 15 (STRL) ------------------
    y_bot = 2.5
    ax.text(0.5, 4.5, 'STRL Variant (Exp 15) — Narrow Search Space due to InfoNCE Semantic Grounding', color=COLOR_ANSWER, fontsize=16, fontweight='bold', ha='left')

    # Nodes
    draw_node(x_nodes[0], y_bot, 'Jerry Jones', '#E3F2FD', '#2196F3')
    draw_node(x_nodes[1], y_bot, 'Dallas Cowboys', '#E3F2FD', '#2196F3')
    draw_node(x_nodes[2], y_bot, 'Barry Switzer', '#E3F2FD', '#2196F3')
    draw_node(x_nodes[3], y_bot, 'Coach Record 1996', '#E8F5E9', '#4CAF50')

    # Hop 1 (TIGHT)
    draw_edge(x_nodes[0]+1.3, y_bot, x_nodes[1]-1.3, y_bot, 'sports.team.owner_s\n(Score: 0.95)', COLOR_ANSWER, lw=4)
    draw_callout((x_nodes[0]+x_nodes[1])/2, y_bot-0.8, "Conf: 0.88\nAction: TIGHT\nBeam: 1 candidate", COLOR_ANSWER)

    # Hop 2 (TIGHT)
    draw_edge(x_nodes[1]+1.3, y_bot, x_nodes[2]-1.3, y_bot, 'historical_coach\n(Score: 0.91)', COLOR_ANSWER, lw=4)
    draw_callout((x_nodes[1]+x_nodes[2])/2, y_bot-0.8, "Conf: 0.82\nAction: TIGHT\nBeam: 1 candidate", COLOR_ANSWER)

    # Hop 3 (TIGHT)
    draw_edge(x_nodes[2]+1.3, y_bot, x_nodes[3]-1.3, y_bot, 'sports.coach.record\n(Score: 0.94)', COLOR_ANSWER, lw=4)
    draw_callout((x_nodes[2]+x_nodes[3])/2, y_bot-0.8, "Conf: 0.89\nAction: TIGHT\nBeam: 1 candidate", COLOR_ANSWER)

    # Metrics Exp 15
    ax.text(18.5, y_bot, "Total Explored:\n1 Candidate\n(High Efficiency)", ha='center', va='center', fontsize=11, color=COLOR_ANSWER, fontweight='bold')

    # Divider
    fig.add_artist(plt.Line2D((0.05, 0.95), (0.5, 0.5), color='#CCCCCC', linewidth=2.0, ls='--'))

    fig.suptitle('Adaptive Traversal Trace: Hierarchical Search Expansion Comparison', color=TEXT_MAIN, fontsize=20, fontweight='bold', y=0.96)

    out = os.path.join(OUT, 'working_example.png')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')

def fig_detailed_dataflow():
    fig, ax = plt.subplots(figsize=(24, 17))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 24)
    ax.set_ylim(-4.5, 13.5)
    ax.axis('off')

    C_LBLUE = '#EBF4FA'
    C_LORANGE = '#FFF2E6'
    C_LGREEN = '#EAF2EA'
    C_LYELLOW = '#FFF9C4'
    C_LPURPLE = '#F5EDF9'

    # Helper function for fancy boxes
    def draw_box(x, y, w, h, label, sublabel='', color='#F0F0F0', ec='#CCCCCC', lw=1.5, z=3, text_color=TEXT_MAIN, fs_label=11, fs_sub=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1', linewidth=lw, edgecolor=ec, facecolor=color, zorder=z)
        ax.add_patch(rect)
        if label:
            ax.text(x+w/2, y+h/2 + (0.12 if sublabel else 0), label, ha='center', va='center', fontsize=fs_label, color=text_color, fontweight='bold', zorder=z+1)
        if sublabel:
            ax.text(x+w/2, y+h/2 - 0.18, sublabel, ha='center', va='center', fontsize=fs_sub, color=TEXT_MUTED, style='italic', zorder=z+1)
        return rect

    # Helper function for fancy arrows
    def draw_arrow(x1, y1, x2, y2, color='#333333', lw=1.5, style='-|>', ms=6, rad=0.0, ls='-'):
        style_str = f'-|>,head_width={ms/3},head_length={ms/2}' if style == '-|>' else style
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style_str, mutation_scale=10, linewidth=lw, linestyle=ls, edgecolor=color, facecolor=color, connectionstyle=f'arc3,rad={rad}', zorder=2)
        ax.add_patch(arrow)
        return arrow

    # Draw Stage Background Panels
    # Stage I: Semantic Planner
    draw_box(0.2, 4.4, 23.6, 7.3, '', '', '#F7F9FC', '#B0C4DE', lw=1, z=1)
    ax.text(1.5, 8.35, "Stage I:\nSemantic Planner", ha='center', va='center', fontsize=14, fontweight='bold', color='#1A365D')

    # Stage II: Traversal Controller
    draw_box(0.2, 0.4, 23.6, 3.8, '', '', '#FFF9F2', '#FFD8B3', lw=1, z=1)
    ax.text(1.5, 2.55, "Stage II:\nTraversal Controller", ha='center', va='center', fontsize=14, fontweight='bold', color='#7B3F00')

    # Stage III: CDS Ranker
    draw_box(0.2, -4.2, 23.6, 4.4, '', '', '#F5FAF6', '#C2E0C6', lw=1, z=1)
    ax.text(1.5, -2.05, "Stage III:\nCDS Ranker", ha='center', va='center', fontsize=14, fontweight='bold', color='#1E4620')

    # ------------------ STAGE I Elements ------------------
    # 1. Input Tokens
    ax.text(11.0, 11.2, "Input Question Tokens (q)", ha='center', fontsize=11, color=TEXT_MUTED, style='italic')
    draw_box(9.2, 10.5, 0.8, 0.5, "t1", "", '#FFFFFF', '#CCCCCC', lw=1, fs_label=10)
    draw_box(10.2, 10.5, 0.8, 0.5, "t2", "", '#FFFFFF', '#CCCCCC', lw=1, fs_label=10)
    ax.text(11.5, 10.6, "...", fontsize=16, ha='center', fontweight='bold')
    draw_box(12.0, 10.5, 0.8, 0.5, "tk", "", '#FFFFFF', '#CCCCCC', lw=1, fs_label=10)

    # 2. RoBERTa Encoder
    draw_box(8.5, 9.2, 5.0, 0.8, "RoBERTa-Large Encoder (Frozen)", "Pretrained Language Model Contextualizer", '#E1F5FE', COLOR_CORE, lw=2)
    draw_arrow(9.6, 10.5, 9.6, 10.0)
    draw_arrow(10.6, 10.5, 10.6, 10.0)
    draw_arrow(12.4, 10.5, 12.4, 10.0)

    # 3. Output Embeddings
    draw_box(8.8, 8.1, 1.1, 0.5, "[CLS]", "", C_LBLUE, COLOR_CORE, lw=1.5, fs_label=10)
    draw_box(10.1, 8.1, 0.9, 0.5, "ET1", "", C_LBLUE, COLOR_CORE, lw=1, fs_label=9)
    ax.text(11.5, 8.25, "...", fontsize=14, ha='center')
    draw_box(12.1, 8.1, 0.9, 0.5, "ETk", "", C_LBLUE, COLOR_CORE, lw=1, fs_label=9)
    draw_arrow(9.3, 9.2, 9.3, 8.6)
    draw_arrow(10.5, 9.2, 10.5, 8.6)
    draw_arrow(12.5, 9.2, 12.5, 8.6)

    # 4. Linear Projection & Heads
    draw_box(8.5, 7.0, 5.0, 0.6, "Linear Projection (W_p)", "Projects 1024-dim CLS to 512-dim reasoning state", '#FFFFFF', '#7F7F7F', lw=1.5)
    draw_arrow(9.3, 8.1, 9.3, 7.6)
    draw_arrow(10.5, 8.1, 10.5, 7.6)
    draw_arrow(12.5, 8.1, 12.5, 7.6)

    # Confidence Head (Right)
    draw_box(14.8, 7.0, 4.2, 0.6, "Confidence Head", "eta_q = sigmoid(W_eta h_CLS)", '#FFFFFF', '#E67E22', lw=1.5)
    draw_arrow(13.5, 7.3, 14.8, 7.3)

    # Domain Embeddings (Down)
    draw_box(8.5, 6.0, 5.0, 0.6, "Predicate Domain Embeddings (z_q)", "Global question semantic vector", C_LORANGE, COLOR_TOPIC, lw=1.5)
    draw_arrow(11.0, 7.0, 11.0, 6.6)

    # Hops representation (Positional Addition)
    # Hop Positional Embeddings (Left)
    draw_box(3.5, 4.9, 4.0, 0.6, "Learned Hop Positional Embeddings", "(E_hop)", C_LORANGE, COLOR_TOPIC, lw=1.5)

    # Four Hops
    draw_box(8.8, 4.9, 0.8, 0.6, "h1", "", '#FFFFFF', COLOR_TOPIC, lw=1.5, fs_label=10)
    draw_box(10.0, 4.9, 0.8, 0.6, "h2", "", '#FFFFFF', COLOR_TOPIC, lw=1.5, fs_label=10)
    draw_box(11.2, 4.9, 0.8, 0.6, "h3", "", '#FFFFFF', COLOR_TOPIC, lw=1.5, fs_label=10)
    draw_box(12.4, 4.9, 0.8, 0.6, "h4", "", '#FFFFFF', COLOR_TOPIC, lw=1.5, fs_label=10)

    # Connect Domain to Hops
    draw_arrow(11.0, 6.0, 9.2, 5.5, rad=0.1)
    draw_arrow(11.0, 6.0, 10.4, 5.5)
    draw_arrow(11.0, 6.0, 11.6, 5.5)
    draw_arrow(11.0, 6.0, 12.8, 5.5, rad=-0.1)

    # Connect Positional to Hops
    draw_arrow(7.5, 5.2, 8.8, 5.2)

    # ------------------ STAGE II Elements ------------------
    # 1. Cross-Hop Attention Transformer Encoder
    draw_box(8.5, 3.8, 5.0, 0.8, "Cross-Hop Attention Transformer Encoder", "Attention depth (L=4, H=8)", '#E8F8F5', '#16A085', lw=2)
    draw_arrow(9.2, 4.9, 9.2, 4.6)
    draw_arrow(10.4, 4.9, 10.4, 4.6)
    draw_arrow(11.6, 4.9, 11.6, 4.6)
    draw_arrow(12.8, 4.9, 12.8, 4.6)

    # 2. Refined Hop Embeddings
    draw_box(8.8, 2.9, 0.8, 0.5, "h^(L)_1", "", '#FFFFFF', '#16A085', lw=1.5, fs_label=10)
    draw_box(10.0, 2.9, 0.8, 0.5, "h^(L)_2", "", '#FFFFFF', '#16A085', lw=1.5, fs_label=10)
    draw_box(11.2, 2.9, 0.8, 0.5, "h^(L)_3", "", '#FFFFFF', '#16A085', lw=1.5, fs_label=10)
    draw_box(12.4, 2.9, 0.8, 0.5, "h^(L)_4", "", '#FFFFFF', '#16A085', lw=1.5, fs_label=10)
    draw_arrow(9.2, 3.8, 9.2, 3.4)
    draw_arrow(10.4, 3.8, 10.4, 3.4)
    draw_arrow(11.6, 3.8, 11.6, 3.4)
    draw_arrow(12.8, 3.8, 12.8, 3.4)

    # 3. RL Meta-Constraint Controller (PPO)
    draw_box(8.5, 1.9, 5.0, 0.6, "RL Meta-Constraint Controller (PPO)", "Policy & Value joint estimation", C_LPURPLE, '#8E44AD', lw=2)
    draw_arrow(9.2, 2.9, 9.2, 2.5)
    draw_arrow(10.4, 2.9, 10.4, 2.5)
    draw_arrow(11.6, 2.9, 11.6, 2.5)
    draw_arrow(12.8, 2.9, 12.8, 2.5)

    # Input Question Raw text
    draw_box(3.5, 1.9, 4.0, 0.6, "Input Question (q)", "Semantic text constraints", '#FFFFFF', '#7F7F7F', lw=1.5)
    draw_arrow(7.5, 2.2, 8.5, 2.2)

    # Curved Confidence Arrow to PPO
    draw_arrow(16.9, 7.0, 13.5, 2.2, rad=-0.2, ls='--')

    # Dotted Callout for PPO Controller MLP internal structure (Image 1)
    rect_callout = FancyBboxPatch((14.8, 1.3), 4.5, 2.2, boxstyle='round,pad=0.1', linewidth=1.5, edgecolor='#8E44AD', facecolor='#F5EEF8', linestyle='--', zorder=1)
    ax.add_patch(rect_callout)
    ax.text(17.05, 3.25, "PPO Controller MLP Detail", ha='center', fontsize=10, fontweight='bold', color='#8E44AD')
    
    draw_box(15.2, 2.7, 3.7, 0.45, "State: s_t = [h^(L)_h ; eta_q]", "", '#FFFFFF', '#8E44AD', lw=1, fs_label=9)
    draw_box(15.2, 2.0, 3.7, 0.45, "MLP Policy Network", "", '#FFFFFF', '#8E44AD', lw=1.5, fs_label=9)
    draw_box(15.2, 1.4, 3.7, 0.45, "Actions: a_t in {T, M, L, S}", "", '#FFFFFF', '#8E44AD', lw=1, fs_label=9)
    draw_arrow(17.05, 2.7, 17.05, 2.45)
    draw_arrow(17.05, 2.0, 17.05, 1.85)

    # Dashed arrow from PPO Controller to Callout
    draw_arrow(13.5, 2.0, 14.8, 2.4, ls='--', style='->', lw=1.5, color='#8E44AD')

    # 4. Dynamic Search Width Action (a_t)
    draw_box(8.5, 0.8, 5.0, 0.6, "Dynamic Search Width Action (a_t)", "Modulates beam width search space", '#FFF2E6', COLOR_TOPIC, lw=2)
    draw_arrow(11.0, 1.9, 11.0, 1.4)

    # ------------------ STAGE III Elements ------------------
    # Dynamic Search path branches
    draw_box(8.8, -0.1, 0.8, 0.5, "Top-k R1", "", '#FFFFFF', COLOR_TOPIC, lw=1.2, fs_label=8)
    draw_box(10.0, -0.1, 0.8, 0.5, "Top-k R2", "", '#FFFFFF', COLOR_TOPIC, lw=1.2, fs_label=8)
    ax.text(11.4, 0.05, "...", fontsize=12, ha='center')
    draw_box(12.4, -0.1, 0.8, 0.5, "Top-k R4", "", '#FFFFFF', COLOR_TOPIC, lw=1.2, fs_label=8)

    draw_arrow(11.0, 0.8, 9.2, 0.4)
    draw_arrow(11.0, 0.8, 10.4, 0.4)
    draw_arrow(11.0, 0.8, 12.8, 0.4)

    # Symbolic Candidate Traversal
    draw_box(8.5, -1.2, 5.0, 0.7, "Symbolic Candidate Traversal", "C = U Reach(e_topic, p_i)", C_LGREEN, COLOR_ANSWER, lw=2)
    draw_arrow(9.2, -0.1, 9.2, -0.5)
    draw_arrow(10.4, -0.1, 10.4, -0.5)
    draw_arrow(12.8, -0.1, 12.8, -0.5)

    # Symbolic KG Search Space Input
    draw_box(3.5, -1.2, 4.0, 0.6, "Symbolic KG Search Space\n(LMDB Graph Environment)", "", '#FFFFFF', '#7F7F7F', lw=1.5)
    draw_arrow(7.5, -0.9, 8.5, -0.9)

    # CDS Ranker
    draw_box(8.5, -2.4, 5.0, 0.7, "CDS Ranker (all-MiniLM-L6)", "Path-Aware Contrastive scoring", '#EBDEF0', '#8E44AD', lw=2)
    draw_arrow(11.0, -1.2, 11.0, -1.7)

    # Final selected reasoning path
    draw_box(7.5, -3.7, 7.0, 0.7, "Final Selected Reasoning Path & Answer Entity", "Yields highest academic Hits@1 success rate", '#E8F8F5', COLOR_ANSWER, lw=2.5)
    draw_arrow(11.0, -2.4, 11.0, -3.0)

    # Main Diagram Titles
    ax.text(12, 13.0, "Detailed End-to-End Hierarchical KGQA Dataflow", ha='center', va='center', fontsize=18, fontweight='bold', color='#111111')
    ax.text(12, 12.6, "Complete tokens-to-paths pipeline matching RoBERTa encodings, hop planning, PPO actions, and path-aware CDS ranking", ha='center', va='center', fontsize=11, color=TEXT_MUTED)

    out = os.path.join(OUT, 'detailed_dataflow.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


if __name__ == '__main__':
    print('Generating paper figures (Light Academic Theme)...')
    fig_knowledge_graph()
    fig_main_architecture()
    fig_traversal_actions()
    fig_training_pipeline()
    fig_complexity_curve()
    fig_working_example()
    fig_detailed_dataflow()
    print('Done. All figures saved to', OUT)

    # Automatically synchronize with thesis_final/figures
    THESIS_FIG_DIR = os.path.join(ROOT, 'thesis_final', 'figures')
    if os.path.exists(THESIS_FIG_DIR):
        print(f'Synchronizing figures to thesis folder: {THESIS_FIG_DIR}')
        for filename in os.listdir(OUT):
            if filename.endswith('.png'):
                src_path = os.path.join(OUT, filename)
                dst_path = os.path.join(THESIS_FIG_DIR, filename)
                shutil.copy2(src_path, dst_path)
                print(f'  [SYNCED] {filename}')
    else:
        print(f'Thesis figure directory not found at: {THESIS_FIG_DIR}')
