"""
AlexNet Feature Map Visualization - PowerPoint Generator
========================================================
Creates a professional, modern 16:9 widescreen presentation deck
incorporating all project analysis, architecture tables, mathematical formulas,
embedded result figures, and viva-voce questions.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette: Modern Dark Slate & Electric Cyan / Indigo
    BG_COLOR = RGBColor(15, 23, 42)        # Slate 900 #0F172A
    CARD_BG = RGBColor(30, 41, 59)        # Slate 800 #1E293B
    BORDER_COLOR = RGBColor(51, 65, 85)   # Slate 700 #334155
    ACCENT_CYAN = RGBColor(56, 189, 248)  # Cyan 400 #38BDF8
    ACCENT_INDIGO = RGBColor(129, 140, 248) # Indigo 400 #818CF8
    ACCENT_AMBER = RGBColor(245, 158, 11) # Amber 500 #F59E0B
    ACCENT_EMERALD = RGBColor(16, 185, 129) # Emerald 500 #10B981
    TEXT_WHITE = RGBColor(248, 250, 252)  # Slate 50
    TEXT_MUTED = RGBColor(148, 163, 184)  # Slate 400

    blank_slide_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        background = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
        )
        background.fill.solid()
        background.fill.fore_color.rgb = BG_COLOR
        background.line.color.rgb = BG_COLOR
        return background

    def add_header(slide, title_text, category_text="ALEXNET FEATURE MAP VISUALIZATION"):
        # Category Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.65))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COLOR):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
        return shape

    # ==========================================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================================
    slide1 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide1)

    # Accent decorative glow bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(0.2), Inches(4.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_CYAN
    bar.line.fill.background()

    # Title & Subtitle Box
    t_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(11.0), Inches(3.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "DEEP LEARNING & COMPUTER VISION"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_CYAN
    p0.space_after = Pt(14)

    p1 = tf.add_paragraph()
    p1.text = "AlexNet Feature Map & Filter Visualization"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p1.space_after = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Demystifying Hierarchical Feature Extraction & Learned Representations in Deep CNNs"
    p2.font.size = Pt(18)
    p2.font.color.rgb = ACCENT_INDIGO

    # Meta card at bottom
    meta_card = add_card(slide1, Inches(1.2), Inches(4.8), Inches(11.0), Inches(1.5))
    m_box = slide1.shapes.add_textbox(Inches(1.4), Inches(4.9), Inches(10.6), Inches(1.3))
    mtf = m_box.text_frame
    
    mp1 = mtf.paragraphs[0]
    mp1.text = "Student Author: Tirumuru Mourya Vardhan"
    mp1.font.size = Pt(15)
    mp1.font.bold = True
    mp1.font.color.rgb = TEXT_WHITE

    mp2 = mtf.add_paragraph()
    mp2.text = "Framework: PyTorch 2.14 & Torchvision | Topic: CNN Interpretability & Feature Progression"
    mp2.font.size = Pt(13)
    mp2.font.color.rgb = TEXT_MUTED

    mp3 = mtf.add_paragraph()
    mp3.text = "Repository: https://github.com/mouryavardhantirumuru/IA-2"
    mp3.font.size = Pt(12)
    mp3.font.color.rgb = ACCENT_CYAN

    # ==========================================================
    # SLIDE 2: INTRODUCTION & MOTIVATION
    # ==========================================================
    slide2 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide2)
    add_header(slide2, "Introduction: Demystifying the CNN 'Black Box'")

    card_w = Inches(3.7)
    card_h = Inches(5.4)
    card_y = Inches(1.5)

    # Card 1: The Problem
    add_card(slide2, Inches(0.8), card_y, card_w, card_h)
    c1_box = slide2.shapes.add_textbox(Inches(1.0), Inches(1.7), card_w - Inches(0.4), card_h - Inches(0.4))
    tf1 = c1_box.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "The Black Box Dilemma"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(10)

    bullets1 = [
        "Deep Neural Networks achieve superhuman classification accuracy but operate as opaque black boxes.",
        "Without internal feature visualization, researchers cannot verify if models learn true semantic cues or dataset biases.",
        "Understanding intermediate layer behavior is essential for trustworthy, explainable AI (XAI)."
    ]
    for b in bullets1:
        pb = tf1.add_paragraph()
        pb.text = f"- {b}"
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # Card 2: The Breakthrough
    add_card(slide2, Inches(4.8), card_y, card_w, card_h)
    c2_box = slide2.shapes.add_textbox(Inches(5.0), Inches(1.7), card_w - Inches(0.4), card_h - Inches(0.4))
    tf2 = c2_box.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "The AlexNet Breakthrough"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(10)

    bullets2 = [
        "AlexNet (Krizhevsky et al., 2012) won ImageNet with a 15.3% error rate (vs. 26.2% runner-up).",
        "Demonstrated that end-to-end deep convolutional representations vastly outperform handcrafted SIFT/HOG descriptors.",
        "Pioneered deep stacking, ReLU non-linearity, GPU acceleration, and dropout regularization."
    ]
    for b in bullets2:
        pb = tf2.add_paragraph()
        pb.text = f"- {b}"
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # Card 3: Project Goals
    add_card(slide2, Inches(8.8), card_y, card_w, card_h)
    c3_box = slide2.shapes.add_textbox(Inches(9.0), Inches(1.7), card_w - Inches(0.4), card_h - Inches(0.4))
    tf3 = c3_box.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "Project Objectives"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_EMERALD
    p.space_after = Pt(10)

    bullets3 = [
        "Extract & visualize feature maps across all 5 convolutional layers using PyTorch forward hooks.",
        "Visualize learned 11x11 RGB filters and correlate with primate visual cortex (V1).",
        "Trace spatial downsampling and receptive field expansion from 11px to 163px.",
        "Quantify activation sparsity (% zero neurons) and activation energy across network depth."
    ]
    for b in bullets3:
        pb = tf3.add_paragraph()
        pb.text = f"- {b}"
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # ==========================================================
    # SLIDE 3: ALEXNET ARCHITECTURAL BLUEPRINT
    # ==========================================================
    slide3 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide3)
    add_header(slide3, "AlexNet Architecture: Mathematical Blueprint & Specifications")

    # Table Card
    add_card(slide3, Inches(0.8), Inches(1.4), Inches(11.7), Inches(4.3))
    
    # Create Table inside slide
    rows = 8
    cols = 8
    table_shape = slide3.shapes.add_table(rows, cols, Inches(1.0), Inches(1.6), Inches(11.3), Inches(3.8))
    table = table_shape.table

    headers = ["Layer", "Type", "Kernel", "Stride", "Pad", "Output Shape", "Parameters", "Receptive Field"]
    table_data = [
        ["Conv1", "Conv2d + ReLU", "11x11", "4", "2", "64 x 55 x 55", "23,296", "11 x 11 px"],
        ["Pool1", "MaxPool2d", "3x3", "2", "0", "64 x 27 x 27", "0", "19 x 19 px"],
        ["Conv2", "Conv2d + ReLU", "5x5", "1", "2", "192 x 27 x 27", "307,392", "51 x 51 px"],
        ["Pool2", "MaxPool2d", "3x3", "2", "0", "192 x 13 x 13", "0", "67 x 67 px"],
        ["Conv3", "Conv2d + ReLU", "3x3", "1", "1", "384 x 13 x 13", "663,936", "99 x 99 px"],
        ["Conv4", "Conv2d + ReLU", "3x3", "1", "1", "256 x 13 x 13", "884,992", "131 x 131 px"],
        ["Conv5", "Conv2d + ReLU", "3x3", "1", "1", "256 x 13 x 13", "590,080", "163 x 163 px"]
    ]

    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN
        p.alignment = PP_ALIGN.CENTER

    for row_idx, row in enumerate(table_data):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 41, 59) if row_idx % 2 == 0 else RGBColor(40, 53, 76)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_WHITE
            p.alignment = PP_ALIGN.CENTER

    # Bottom Formula Card
    add_card(slide3, Inches(0.8), Inches(5.9), Inches(11.7), Inches(1.1))
    f_box = slide3.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.3), Inches(0.95))
    tf_f = f_box.text_frame
    tf_f.word_wrap = True
    pf1 = tf_f.paragraphs[0]
    pf1.text = "Mathematical Receptive Field Recurrence Formula:"
    pf1.font.bold = True
    pf1.font.size = Pt(12)
    pf1.font.color.rgb = ACCENT_EMERALD

    pf2 = tf_f.add_paragraph()
    pf2.text = "RF_l = RF_{l-1} + (k_l - 1) * J_{l-1}    where Cumulative Feature Jump J_l = J_{l-1} * s_l  (Initial: RF_0=1, J_0=1)"
    pf2.font.size = Pt(11)
    pf2.font.color.rgb = TEXT_WHITE

    # ==========================================================
    # SLIDE 4: METHODOLOGY - PYTORCH HOOKS PIPELINE
    # ==========================================================
    slide4 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide4)
    add_header(slide4, "Methodology: Intercepting Activations via PyTorch Forward Hooks")

    # Step 1 Card
    add_card(slide4, Inches(0.8), Inches(1.5), Inches(3.7), Inches(5.3))
    s1_box = slide4.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(3.3), Inches(4.9))
    ts1 = s1_box.text_frame
    ts1.word_wrap = True
    p = ts1.paragraphs[0]
    p.text = "1. Model & Hook Setup"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(10)

    b_step1 = [
        "In PyTorch, intermediate layer tensors are deleted during forward execution to minimize memory.",
        "Forward hooks are non-invasive listeners attached to specific layers:",
        "layer.register_forward_hook(hook_fn)",
        "Cached layers: features[0] (Conv1), features[3] (Conv2), features[6] (Conv3), features[8] (Conv4), features[10] (Conv5)."
    ]
    for b in b_step1:
        pb = ts1.add_paragraph()
        pb.text = f"- {b}"
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # Step 2 Card
    add_card(slide4, Inches(4.8), Inches(1.5), Inches(3.7), Inches(5.3))
    s2_box = slide4.shapes.add_textbox(Inches(5.0), Inches(1.7), Inches(3.3), Inches(4.9))
    ts2 = s2_box.text_frame
    ts2.word_wrap = True
    p = ts2.paragraphs[0]
    p.text = "2. Input Preprocessing"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO
    p.space_after = Pt(10)

    b_step2 = [
        "Images resized to 256px and Center Cropped to 224x224 RGB.",
        "Normalized using ImageNet statistics:",
        "mean = [0.485, 0.456, 0.406]",
        "std = [0.229, 0.224, 0.225]",
        "Guarantees that input distribution matches the pretrained weights' training domain."
    ]
    for b in b_step2:
        pb = ts2.add_paragraph()
        pb.text = f"- {b}"
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # Step 3 Card
    add_card(slide4, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.3))
    s3_box = slide4.shapes.add_textbox(Inches(9.0), Inches(1.7), Inches(3.3), Inches(4.9))
    ts3 = s3_box.text_frame
    ts3.word_wrap = True
    p = ts3.paragraphs[0]
    p.text = "3. Extraction & Analytics"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_EMERALD
    p.space_after = Pt(10)

    b_step3 = [
        "Tensor Detachment: Output tensors cloned and detached from autograd graph.",
        "Energy Sorting: Channels ranked by L1 activation energy sum.",
        "Spatial Attention Mapping: Mean layer activations bilinearly interpolated onto original image.",
        "Sparsity Analytics: Percentage of exact zero neurons measured per layer."
    ]
    for b in b_step3:
        pb = ts3.add_paragraph()
        pb.text = f"- {b}"
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # ==========================================================
    # SLIDE 5: CONV1 LEARNED FILTERS
    # ==========================================================
    slide5 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide5)
    add_header(slide5, "Conv1 Learned Spatial Weight Filters (64 Filters, 11x11x3)")

    # Left: Image Card
    add_card(slide5, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.5))
    f_path = "results/conv1_filters_visualization.png"
    if os.path.exists(f_path):
        slide5.shapes.add_picture(f_path, Inches(1.0), Inches(1.6), Inches(5.4), Inches(5.1))

    # Right: Analysis Card
    add_card(slide5, Inches(6.9), Inches(1.4), Inches(5.6), Inches(5.5))
    f_text_box = slide5.shapes.add_textbox(Inches(7.1), Inches(1.6), Inches(5.2), Inches(5.1))
    tff = f_text_box.text_frame
    tff.word_wrap = True

    p = tff.paragraphs[0]
    p.text = "Key Observations & Neurological Parallels"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(12)

    filter_bullets = [
        "Gabor-Like Directional Edges: Clear alternating bright and dark stripes oriented at 0 deg, 45 deg, 90 deg, and 135 deg to capture directional boundaries.",
        "Frequency Wavelets: Filters span from coarse, low-frequency transitions to rapid, high-frequency spatial ripples.",
        "Color Opponency: Specialized color antagonist pairs (e.g., Blue-Yellow, Red-Cyan, Green-Magenta) emerge spontaneously.",
        "Biological Parallels: These learned mathematical weights mirror the receptive fields of simple cells in the mammalian primary visual cortex (V1) discovered by Hubel & Wiesel.",
        "Independent Discovery: Demonstrates that gradient descent naturally rediscovers optimal representations for natural image statistics."
    ]
    for b in filter_bullets:
        pb = tff.add_paragraph()
        pb.text = f"- {b}"
        pb.font.size = Pt(11)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # ==========================================================
    # SLIDE 6: CONV1 & CONV2 FEATURE MAPS
    # ==========================================================
    slide6 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide6)
    add_header(slide6, "Early Layers: Low-Level Edges & Intermediate Textures")

    # Left: Conv1 Feature Maps
    add_card(slide6, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.5))
    c1_path = "results/conv1_feature_maps.png"
    if os.path.exists(c1_path):
        slide6.shapes.add_picture(c1_path, Inches(1.0), Inches(1.6), Inches(5.3), Inches(4.2))
    
    c1_desc = slide6.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(5.3), Inches(0.9))
    tc1 = c1_desc.text_frame
    tc1.word_wrap = True
    p = tc1.paragraphs[0]
    p.text = "Conv1 (55x55, 64 channels, RF: 11px): Retains raw pixel-level geometry. Channels respond to directional silhouettes, high-contrast outlines, and lighting gradients."
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_MUTED

    # Right: Conv2 Feature Maps
    add_card(slide6, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.5))
    c2_path = "results/conv2_feature_maps.png"
    if os.path.exists(c2_path):
        slide6.shapes.add_picture(c2_path, Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.2))

    c2_desc = slide6.shapes.add_textbox(Inches(7.0), Inches(5.9), Inches(5.3), Inches(0.9))
    tc2 = c2_desc.text_frame
    tc2.word_wrap = True
    p = tc2.paragraphs[0]
    p.text = "Conv2 (27x27, 192 channels, RF: 51px): Max-pooling halves resolution. Channels combine simple edges into corners, fur textures, concentric curves, and mesh patterns."
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_MUTED

    # ==========================================================
    # SLIDE 7: DEEPER LAYERS: CONV3, CONV4 & CONV5
    # ==========================================================
    slide7 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide7)
    add_header(slide7, "Deeper Layers: Transition from Parts to Semantic Abstractions")

    # Card 1: Conv3
    add_card(slide7, Inches(0.8), Inches(1.4), Inches(3.7), Inches(5.5))
    c3_path = "results/conv3_feature_maps.png"
    if os.path.exists(c3_path):
        slide7.shapes.add_picture(c3_path, Inches(0.95), Inches(1.55), Inches(3.4), Inches(3.4))
    c3_desc = slide7.shapes.add_textbox(Inches(0.95), Inches(5.1), Inches(3.4), Inches(1.6))
    tc3 = c3_desc.text_frame
    tc3.word_wrap = True
    p = tc3.paragraphs[0]
    p.text = "Conv3 (13x13, 384ch, RF: 99px)"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_CYAN
    p2 = tc3.add_paragraph()
    p2.text = "Structural Motifs: Discards exact pixel coordinates. Encodes translation-invariant complex textures and structural patterns."
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_MUTED

    # Card 2: Conv4
    add_card(slide7, Inches(4.8), Inches(1.4), Inches(3.7), Inches(5.5))
    c4_path = "results/conv4_feature_maps.png"
    if os.path.exists(c4_path):
        slide7.shapes.add_picture(c4_path, Inches(4.95), Inches(1.55), Inches(3.4), Inches(3.4))
    c4_desc = slide7.shapes.add_textbox(Inches(4.95), Inches(5.1), Inches(3.4), Inches(1.6))
    tc4 = c4_desc.text_frame
    tc4.word_wrap = True
    p = tc4.paragraphs[0]
    p.text = "Conv4 (13x13, 256ch, RF: 131px)"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_INDIGO
    p2 = tc4.add_paragraph()
    p2.text = "Object Sub-Parts: Specific channels fire on recognizable sub-components such as snouts, eyes, paw outlines, and fur clumps."
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_MUTED

    # Card 3: Conv5
    add_card(slide7, Inches(8.8), Inches(1.4), Inches(3.7), Inches(5.5))
    c5_path = "results/conv5_feature_maps.png"
    if os.path.exists(c5_path):
        slide7.shapes.add_picture(c5_path, Inches(8.95), Inches(1.55), Inches(3.4), Inches(3.4))
    c5_desc = slide7.shapes.add_textbox(Inches(8.95), Inches(5.1), Inches(3.4), Inches(1.6))
    tc5 = c5_desc.text_frame
    tc5.word_wrap = True
    p = tc5.paragraphs[0]
    p.text = "Conv5 (13x13, 256ch, RF: 163px)"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_EMERALD
    p2 = tc5.add_paragraph()
    p2.text = "Semantic Selectivity: Extreme activation sparsity (~89%). Channels respond selectively to the entire holistic object category."
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_MUTED

    # ==========================================================
    # SLIDE 8: HIERARCHICAL REPRESENTATION PROGRESSION
    # ==========================================================
    slide8 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide8)
    add_header(slide8, "Cross-Layer Hierarchical Representation Progression")

    # Image Card
    add_card(slide8, Inches(0.8), Inches(1.4), Inches(11.7), Inches(4.3))
    prog_path = "results/layer_progression_comparison.png"
    if os.path.exists(prog_path):
        slide8.shapes.add_picture(prog_path, Inches(1.0), Inches(1.5), Inches(11.3), Inches(4.0))

    # Bottom Explanation
    add_card(slide8, Inches(0.8), Inches(5.9), Inches(11.7), Inches(1.1))
    prog_box = slide8.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.3), Inches(0.95))
    tpr = prog_box.text_frame
    tpr.word_wrap = True
    p = tpr.paragraphs[0]
    p.text = "The Fundamental Law of Deep Convolutional Hierarchies:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_CYAN
    p2 = tpr.add_paragraph()
    p2.text = "Input Pixels (224x224) -> High Spatial Resolution, Low Semantic Meaning  ===>  Conv5 Activations (13x13) -> Low Spatial Resolution, High Semantic Invariance."
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_WHITE

    # ==========================================================
    # SLIDE 9: SPATIAL ATTENTION HEATMAP OVERLAYS
    # ==========================================================
    slide9 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide9)
    add_header(slide9, "Spatial Activation Heatmap Overlays: Localizing Network Attention")

    # Image Card
    add_card(slide9, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.6))
    heat_path = "results/activation_heatmaps_overlay.png"
    if os.path.exists(heat_path):
        slide9.shapes.add_picture(heat_path, Inches(1.0), Inches(1.5), Inches(11.3), Inches(3.4))

    # Bottom 3 Insight Cards
    b_y = Inches(5.2)
    b_h = Inches(1.8)

    add_card(slide9, Inches(0.8), b_y, Inches(3.7), b_h)
    tb1 = slide9.shapes.add_textbox(Inches(0.9), b_y + Inches(0.1), Inches(3.5), b_h - Inches(0.2)).text_frame
    tb1.word_wrap = True
    p = tb1.paragraphs[0]
    p.text = "Early Layers: Uniform Coverage"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_CYAN
    p = tb1.add_paragraph()
    p.text = "Conv1 and Conv2 attend evenly across all textured areas, borders, and foreground-background transitions."
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide9, Inches(4.8), b_y, Inches(3.7), b_h)
    tb2 = slide9.shapes.add_textbox(Inches(4.9), b_y + Inches(0.1), Inches(3.5), b_h - Inches(0.2)).text_frame
    tb2.word_wrap = True
    p = tb2.paragraphs[0]
    p.text = "Mid Layers: Target Localization"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_INDIGO
    p = tb2.add_paragraph()
    p.text = "Conv3 and Conv4 begin filtering out background noise, clustering energy tightly onto the primary subject."
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide9, Inches(8.8), b_y, Inches(3.7), b_h)
    tb3 = slide9.shapes.add_textbox(Inches(8.9), b_y + Inches(0.1), Inches(3.5), b_h - Inches(0.2)).text_frame
    tb3.word_wrap = True
    p = tb3.paragraphs[0]
    p.text = "Deep Layer: Semantic Saliency"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_EMERALD
    p = tb3.add_paragraph()
    p.text = "Conv5 confines its peak activation energy directly to the most discriminative anatomical regions of the dog (face/head)."
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_WHITE

    # ==========================================================
    # SLIDE 10: QUANTITATIVE ACTIVATION & SPARSITY METRICS
    # ==========================================================
    slide10 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide10)
    add_header(slide10, "Quantitative Activation Analytics: Sparsity & Energy Dynamics")

    # Left: Stats Image
    add_card(slide10, Inches(0.8), Inches(1.4), Inches(6.5), Inches(5.5))
    stat_path = "results/activation_statistics.png"
    if os.path.exists(stat_path):
        slide10.shapes.add_picture(stat_path, Inches(1.0), Inches(1.6), Inches(6.1), Inches(5.1))

    # Right: Data Table & Analysis
    add_card(slide10, Inches(7.5), Inches(1.4), Inches(5.0), Inches(5.5))
    st_box = slide10.shapes.add_textbox(Inches(7.7), Inches(1.6), Inches(4.6), Inches(5.1))
    tst = st_box.text_frame
    tst.word_wrap = True

    p = tst.paragraphs[0]
    p.text = "Empirical Layer Metrics (Dog Sample)"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(10)

    stats_summary = [
        "Conv1: Sparsity = 60.08% | Energy = 0.377 | Max = 19.65",
        "Conv2: Sparsity = 80.58% | Energy = 0.437 | Max = 33.81",
        "Conv3: Sparsity = 78.82% | Energy = 0.623 | Max = 35.42",
        "Conv4: Sparsity = 82.04% | Energy = 0.529 | Max = 30.65",
        "Conv5: Sparsity = 88.98% | Energy = 0.280 | Max = 29.68"
    ]
    for s in stats_summary:
        pb = tst.add_paragraph()
        pb.text = s
        pb.font.size = Pt(11)
        pb.font.bold = True
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(4)

    p_div = tst.add_paragraph()
    p_div.text = "\nCore Findings:"
    p_div.font.bold = True
    p_div.font.size = Pt(13)
    p_div.font.color.rgb = ACCENT_AMBER
    p_div.space_after = Pt(6)

    findings = [
        "Monotonic Sparsity Growth: Nearly 89% of Conv5 neurons remain at exact zero for any single input due to ReLU clamping.",
        "Selective Firing: Deep neurons only trigger when a precise, highly complex conjunction of sub-features is detected.",
        "Energy Efficiency: High sparsity enables fast sparse inference and reduces redundant channel interference."
    ]
    for f in findings:
        pb = tst.add_paragraph()
        pb.text = f"- {f}"
        pb.font.size = Pt(10)
        pb.font.color.rgb = TEXT_MUTED
        pb.space_after = Pt(4)

    # ==========================================================
    # SLIDE 11: IMPLEMENTATION & LIVE CLI DEMO
    # ==========================================================
    slide11 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide11)
    add_header(slide11, "Software Architecture & Live Demonstration CLI")

    # Left: Output Screenshot
    add_card(slide11, Inches(0.8), Inches(1.4), Inches(6.8), Inches(5.5))
    out_path = "screenshots/output.png"
    if os.path.exists(out_path):
        slide11.shapes.add_picture(out_path, Inches(1.0), Inches(1.6), Inches(6.4), Inches(5.1))

    # Right: CLI & Code Features
    add_card(slide11, Inches(7.8), Inches(1.4), Inches(4.7), Inches(5.5))
    cli_box = slide11.shapes.add_textbox(Inches(8.0), Inches(1.6), Inches(4.3), Inches(5.1))
    tcli = cli_box.text_frame
    tcli.word_wrap = True

    p = tcli.paragraphs[0]
    p.text = "Modular & Extensible CLI"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(10)

    cli_items = [
        "Fully Parametrized CLI:\npython src/alexnet_feature_visualization.py --image <path> --output_dir <path>",
        "Live Demonstration Ready: Supports seamless testing with any benchmark image (dog, butterfly, baboon, car, fruits).",
        "Jupyter Walkthrough:\nInteractive experimentation notebook included in notebooks/ directory.",
        "Zero Hardcoded Shortcuts: Pure PyTorch dynamic forward hooks with clean resource detachment.",
        "Git Authenticity: Step-by-step verified commits tracked in GitHub repository."
    ]
    for ci in cli_items:
        pb = tcli.add_paragraph()
        pb.text = f"- {ci}"
        pb.font.size = Pt(11)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # ==========================================================
    # SLIDE 12: VIVA-VOCE QUESTIONS & CONCLUSION
    # ==========================================================
    slide12 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide12)
    add_header(slide12, "Viva-Voce Defense & Conclusion")

    # Left: Top Viva Q&A
    add_card(slide12, Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.5))
    viva_box = slide12.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.6), Inches(5.1))
    tv = viva_box.text_frame
    tv.word_wrap = True

    p = tv.paragraphs[0]
    p.text = "Critical Viva-Voce Questions"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(8)

    vivas = [
        ("Q1: Why 11x11 kernel in Conv1 instead of 3x3?",
         "AlexNet needed to rapidly downsample 224x224 images down to 55x55 to fit memory on dual 3GB GPUs. Later networks proved two 3x3 layers give equivalent 5x5 coverage with fewer weights."),
        ("Q2: Why use PyTorch forward hooks?",
         "PyTorch deletes intermediate tensors during execution. Forward hooks intercept and clone activation maps on-the-fly without altering the neural network architecture."),
        ("Q3: Why do learned filters match Gabor filters?",
         "Step edges and oriented luminance transitions dominate natural images. Gradient descent naturally converges to the optimal mathematical basis for these statistics.")
    ]
    for q, a in vivas:
        pq = tv.add_paragraph()
        pq.text = q
        pq.font.bold = True
        pq.font.size = Pt(11)
        pq.font.color.rgb = ACCENT_CYAN
        pa = tv.add_paragraph()
        pa.text = a
        pa.font.size = Pt(10)
        pa.font.color.rgb = TEXT_WHITE
        pa.space_after = Pt(6)

    # Right: Project Summary & Wrap-up
    add_card(slide12, Inches(7.1), Inches(1.4), Inches(5.4), Inches(5.5))
    sum_box = slide12.shapes.add_textbox(Inches(7.3), Inches(1.6), Inches(5.0), Inches(5.1))
    tsm = sum_box.text_frame
    tsm.word_wrap = True

    p = tsm.paragraphs[0]
    p.text = "Key Takeaways"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = ACCENT_EMERALD
    p.space_after = Pt(8)

    takeaways = [
        "Hierarchical Feature Extraction: Successfully demonstrated the progression from low-level edges (Conv1) to high-level semantic abstractions (Conv5).",
        "Receptive Field Expansion: Verified theoretical expansion from 11px to 163px.",
        "Sparsity Verification: Confirmed ReLU-driven sparsity increasing from 60% up to 89% in deep layers.",
        "Reproducibility: End-to-end Python pipeline, interactive Jupyter notebook, and verified Git commit history ready for evaluation."
    ]
    for t in takeaways:
        pb = tsm.add_paragraph()
        pb.text = f"- {t}"
        pb.font.size = Pt(11)
        pb.font.color.rgb = TEXT_WHITE
        pb.space_after = Pt(8)

    # Save presentation
    output_filename = "AlexNet_Feature_Map_Visualization_Presentation.pptx"
    prs.save(output_filename)
    print(f"[SUCCESS] Presentation generated: {output_filename}")

if __name__ == "__main__":
    create_presentation()
