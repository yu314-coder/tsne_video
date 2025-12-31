"""
Manim Animation: k-distance in t-SNE Embeddings
Complete visualization of the mathematical theory from Yu, Yao-Hsing (2025)

To render:
    manim -pql tsne_kdistance_full.py TSNEPresentation
    manim -pqh tsne_kdistance_full.py TSNEPresentation  # High quality
    manim -pqk tsne_kdistance_full.py TSNEPresentation  # 4K quality
"""

from manim import *
import numpy as np

# ============================================================================
# CONFIGURATION & COLORS - TECH/FUTURISTIC THEME
# ============================================================================

config.background_color = BLACK

# Tech Color Palette (optimized for black background)
DARK_BLUE = "#00d4ff"      # Cyan blue - tech accent
DARK_ORANGE = "#ff6b35"    # Vibrant orange
DARK_GREEN = "#00ff88"     # Neon green
DARK_RED = "#ff3366"       # Hot pink red
DARK_PURPLE = "#b794f6"    # Purple
TEXT_COLOR = "#e0e0e0"     # Light gray text
HIGHLIGHT_COLOR = "#ff3366" # Hot pink
SECONDARY_COLOR = "#00d4ff" # Cyan
LIGHT_GRAY = "#808080"     # Medium gray
TECH_CYAN = "#00ffff"      # Bright cyan
TECH_MAGENTA = "#ff00ff"   # Bright magenta
TECH_YELLOW = "#ffff00"    # Bright yellow


# ============================================================================
# HELPER FUNCTIONS TO PREVENT TEXT OVERFLOW
# ============================================================================

def safe_text(text, font_size=28, max_width_ratio=0.9, **kwargs):
    """Create Text that automatically fits within frame"""
    t = Text(text, font_size=font_size, **kwargs)
    max_w = config.frame_width * max_width_ratio
    if t.get_width() > max_w:
        t.set_width(max_w)
    return t

def safe_mathtex(latex, font_size=32, max_width_ratio=0.85, **kwargs):
    """Create MathTex that automatically fits within frame"""
    m = MathTex(latex, font_size=font_size, **kwargs)
    max_w = config.frame_width * max_width_ratio
    if m.get_width() > max_w:
        m.set_width(max_w)
    return m

def constrain_width(mobject, max_width_ratio=0.9):
    """Constrain existing mobject to fit within frame"""
    max_w = config.frame_width * max_width_ratio
    if mobject.get_width() > max_w:
        mobject.set_width(max_w)
    return mobject


# ============================================================================
# COMPLETE PRESENTATION IN ONE CLASS
# ============================================================================

class TSNEPresentation(ThreeDScene):
    def construct(self):
        # ====================================================================
        # SCENE 1: TITLE
        # ====================================================================
        self.scene_01_title()
        
        # ====================================================================
        # SCENE 2: WHAT IS t-SNE?
        # ====================================================================
        self.scene_02_what_is_tsne()
        
        # ====================================================================
        # SCENE 3: t-SNE MATHEMATICS
        # ====================================================================
        self.scene_03_tsne_math()
        
        # ====================================================================
        # SCENE 4: k-DISTANCE DEFINITION
        # ====================================================================
        self.scene_04_kdistance()

        # VIDEO ENDS HERE - As requested by user

    # ========================================================================
    # SCENE 1: TITLE
    # ========================================================================

    def scene_01_title(self):
        # Main title
        title = Text(
            "k-distance in t-SNE Embeddings",
            font_size=56,
            color=TEXT_COLOR,
            weight=BOLD
        )
        
        # Subtitle
        subtitle = Text(
            "A Mathematical Analysis of Geometric Transformations",
            font_size=32,
            color=DARK_BLUE
        ).next_to(title, DOWN, buff=0.5)
        
        # Author name and date removed for privacy
        
        # Animate
        self.play(
            Write(title),
            run_time=2
        )
        self.wait(0.5)
        
        self.play(
            FadeIn(subtitle, shift=UP),
            run_time=1.5
        )
        self.wait(2.5)  # Extended wait since no author/date to show
        
        # Fade out
        self.play(
            *[FadeOut(mob) for mob in [title, subtitle]],
            run_time=1
        )
    
    # ========================================================================
    # SCENE 2: WHAT IS t-SNE?
    # ========================================================================
    
    def scene_02_what_is_tsne(self):
        # ====================================================================
        # PRE-LOAD MNIST (before animation starts)
        # ====================================================================
        
        print("Pre-loading MNIST data...")
        try:
            import requests
            import gzip
            import struct
            from sklearn.manifold import TSNE
            
            # Download silently
            base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
            
            response = requests.get(base_url + "train-images-idx3-ubyte.gz", timeout=60)
            response.raise_for_status()
            images_data = gzip.decompress(response.content)
            
            response_labels = requests.get(base_url + "train-labels-idx1-ubyte.gz", timeout=60)
            response_labels.raise_for_status()
            labels_data = gzip.decompress(response_labels.content)
            
            _, num_images, rows, cols = struct.unpack('>IIII', images_data[:16])
            images = np.frombuffer(images_data[16:], dtype=np.uint8).reshape(num_images, rows, cols)
            labels = np.frombuffer(labels_data[8:], dtype=np.uint8)
            
            # Select 16 samples (2 per digit for 8 digits) for cleaner visualization
            selected_images = []
            selected_labels = []
            for digit in range(8):  # Use digits 0-7 (16 total samples)
                indices = np.where(labels == digit)[0][:2]
                selected_images.extend(images[indices])
                selected_labels.extend(labels[indices])
            
            selected_images = np.array(selected_images)
            selected_labels = np.array(selected_labels)
            
            # Pre-compute t-SNE (silently)
            print("Pre-computing t-SNE...")
            X_flat = selected_images.reshape(len(selected_images), -1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=1000)
            X_embedded = tsne.fit_transform(X_flat)
            
            # Normalize to screen coordinates
            x_min, x_max = X_embedded[:, 0].min(), X_embedded[:, 0].max()
            y_min, y_max = X_embedded[:, 1].min(), X_embedded[:, 1].max()
            X_scaled = np.zeros_like(X_embedded)
            X_scaled[:, 0] = -5.5 + 11 * (X_embedded[:, 0] - x_min) / (x_max - x_min)
            X_scaled[:, 1] = -3.0 + 4.6 * (X_embedded[:, 1] - y_min) / (y_max - y_min)  # Moved down from -2.3
            
            mnist_loaded = True
            print("✓ Data ready!")
            
        except Exception as e:
            print(f"Failed to load MNIST: {e}")
            mnist_loaded = False
        
        if not mnist_loaded:
            error_msg = Text("MNIST download failed", font_size=24, color=DARK_RED)
            error_msg.move_to(ORIGIN)
            self.add_fixed_in_frame_mobjects(error_msg)
            self.play(FadeIn(error_msg))
            self.wait(2)
            self.play(FadeOut(error_msg))
            return
        
        # ====================================================================
        # START ANIMATION - Title
        # ====================================================================
        
        title = Text("What is t-SNE?", font_size=64, color=DARK_BLUE, weight=BOLD)
        title.move_to(UP * 3.3)

        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait(2.1)  # Extended for narration segment 1

        # ====================================================================
        # Show "Example: MNIST dataset" BEFORE Step 1
        # ====================================================================

        dataset_label = Text("Example: MNIST dataset", font_size=60, color=TECH_CYAN, weight=BOLD)
        dataset_label.move_to(ORIGIN)
        
        self.add_fixed_in_frame_mobjects(dataset_label)
        self.play(FadeIn(dataset_label, scale=1.1))
        self.wait(1.5)
        self.play(FadeOut(dataset_label))
        self.wait(0.3)
        
        # ====================================================================
        # STEP 1: Show Real MNIST Images
        # ====================================================================

        step_title = Text("Step 1: Handwritten Digits", font_size=26, color=DARK_ORANGE, weight=BOLD)
        step_title.move_to(UP * 2.5)
        
        self.add_fixed_in_frame_mobjects(step_title)
        self.play(FadeIn(step_title))

        # Use WHITE color for all digits
        def create_mnist_visual(img_array, color, scale=0.05):
            """Create actual MNIST pixel visualization with WHITE color"""
            pixels = VGroup()
            img_norm = img_array / 255.0

            # Render actual pixels with HIGHER opacity for visibility
            for i in range(0, 28, 2):  # Every 2nd pixel for performance
                for j in range(0, 28, 2):
                    val = img_norm[i, j]
                    if val > 0.1:  # Lower threshold to show more pixels
                        pixel = Square(
                            side_length=scale,
                            fill_color=color,
                            fill_opacity=min(val * 1.2, 1.0),  # Increased brightness: 1.2x
                            stroke_width=0.5,  # Add slight stroke for definition
                            stroke_color=color,
                            stroke_opacity=0.8
                        ).shift(np.array([
                            (j - 14) * scale,
                            (14 - i) * scale,
                            0
                        ]))
                        pixels.add(pixel)
            return pixels

        # Create 4×4 grid of 16 real MNIST images (WHITE color)
        all_images = VGroup()
        all_image_data = []  # Store actual image arrays

        grid_spacing = 1.5  # Increased spacing for larger matrices/vectors
        # Center the 4x4 grid
        start_x = -2.25  # Centered horizontally
        start_y = 1.5    # Positioned to avoid both title and bottom edge

        for idx in range(16):
            row = idx // 4  # 4×4 grid
            col = idx % 4

            img = selected_images[idx]
            label = selected_labels[idx]
            all_image_data.append(img)

            # Create actual pixel visualization (WHITE color)
            mnist_img = create_mnist_visual(img, WHITE)
            mnist_img.move_to(np.array([
                start_x + col * grid_spacing,
                start_y - row * grid_spacing,
                0
            ]))
            all_images.add(mnist_img)
        
        # Show all 16 real MNIST images
        self.play(*[FadeIn(img, scale=0.4) for img in all_images], run_time=2)
        self.wait(1.5)
        
        # ====================================================================
        # STEP 2: Transform to REAL Matrices (show actual pixel values)
        # ====================================================================
        
        step_title2 = Text("Step 2: Each → 28×28 Matrix (actual values)", font_size=24, color=DARK_ORANGE, weight=BOLD)
        step_title2.move_to(UP * 2.5)
        
        self.add_fixed_in_frame_mobjects(step_title2)
        self.play(FadeOut(step_title), FadeIn(step_title2))
        
        # Create matrix representations with REAL pixel values - WHITE COLOR
        all_matrices = VGroup()

        for idx in range(16):
            img = all_image_data[idx]
            label = selected_labels[idx]

            # Extract sample of real pixel values (center region for visibility)
            real_values = img[12:15, 12:15]  # 3×3 sample from center

            # Create matrix with actual values - WHITE color
            matrix_rows = []
            for i in range(3):
                row_values = []
                for j in range(3):
                    val_text = Text(
                        f"{int(real_values[i, j]):3d}",
                        font_size=11,
                        color=WHITE,
                        weight=BOLD
                    )
                    row_values.append(val_text)
                row = VGroup(*row_values).arrange(RIGHT, buff=0.08)
                matrix_rows.append(row)

            matrix_content = VGroup(*matrix_rows).arrange(DOWN, buff=0.05)

            # Add brackets
            left_bracket = Text("⎡", font_size=22, color=WHITE, weight=BOLD)
            right_bracket = Text("⎤", font_size=22, color=WHITE, weight=BOLD)
            left_bracket.next_to(matrix_content, LEFT, buff=0.08)
            right_bracket.next_to(matrix_content, RIGHT, buff=0.08)

            matrix = VGroup(left_bracket, matrix_content, right_bracket)

            # Position at same location as original image
            matrix.move_to(all_images[idx].get_center())
            all_matrices.add(matrix)
        
        # Fade out images and fade in matrices
        self.play(
            *[FadeOut(all_images[i]) for i in range(16)],
            run_time=0.8
        )
        self.add_fixed_in_frame_mobjects(*all_matrices)
        self.play(
            *[FadeIn(all_matrices[i]) for i in range(16)],
            run_time=1.5
        )
        self.wait(1.5)
        
        # ====================================================================
        # STEP 3: Flatten to REAL Vectors (show actual values)
        # ====================================================================
        
        step_title3 = Text("Step 3: Flatten → 784×1 Vector", font_size=24, color=DARK_ORANGE, weight=BOLD)
        step_title3.move_to(UP * 2.5)
        
        self.add_fixed_in_frame_mobjects(step_title3)
        self.play(FadeOut(step_title2), FadeIn(step_title3))
        
        # Create vector representations with REAL flattened values - WHITE COLOR
        all_vectors = VGroup()

        for idx in range(16):
            img = all_image_data[idx]
            label = selected_labels[idx]

            # Flatten and get values from middle where digit data exists
            flattened = img.flatten()
            # Show only 3 values from middle rows where digit pixels are
            sample_indices = [300, 350, 400]  # Sample from middle rows
            sample_values = [flattened[i] for i in sample_indices]

            # Create vertical vector display - WHITE color
            vector_rows = []
            for val in sample_values:
                val_text = Text(
                    f"{int(val):3d}",
                    font_size=10,
                    color=WHITE,
                    weight=BOLD
                )
                vector_rows.append(val_text)

            dots = Text("⋮", font_size=14, color=WHITE, weight=BOLD)
            vector_rows.append(dots)

            vector_content = VGroup(*vector_rows).arrange(DOWN, buff=0.03)

            # Add brackets
            left_bracket = Text("[", font_size=18, color=WHITE, weight=BOLD)
            right_bracket = Text("]", font_size=18, color=WHITE, weight=BOLD)
            left_bracket.next_to(vector_content, LEFT, buff=0.05)
            right_bracket.next_to(vector_content, RIGHT, buff=0.05)

            vector = VGroup(left_bracket, vector_content, right_bracket)

            vector.move_to(all_matrices[idx].get_center())
            all_vectors.add(vector)
        
        # Fade out matrices and fade in vectors
        self.play(
            *[FadeOut(all_matrices[i]) for i in range(16)],
            run_time=0.8
        )
        self.add_fixed_in_frame_mobjects(*all_vectors)
        self.play(
            *[FadeIn(all_vectors[i]) for i in range(16)],
            run_time=1.5
        )
        self.wait(3.14)  # Extended for narration segment 3

        # ====================================================================
        # STEP 4: Combine into Data Matrix
        # ====================================================================

        step_title_matrix = Text("Step 4: Combine into Data Matrix", font_size=26, color=DARK_ORANGE, weight=BOLD)
        step_title_matrix.move_to(UP * 2.5)

        self.add_fixed_in_frame_mobjects(step_title_matrix)
        self.play(FadeOut(step_title3), FadeIn(step_title_matrix))

        # Create 784×16 data matrix visualization (LARGER to take up more space)
        # Each column is one flattened 784×1 image vector
        # Show more rows with larger font

        matrix_entries = []

        # Column spacing - fixed positions for proper alignment
        col_spacing = 0.42  # Fixed spacing between columns (increased for readability)
        start_x = -3.3  # Starting x position (adjusted for 16 columns with wider spacing)

        # Show rows from middle of images where actual digit data exists (rows 300-311)
        # First 12 rows of the flattened vector are blank padding in MNIST
        for row_idx in range(300, 312):
            row_values = []
            for col_idx in range(16):
                img = all_image_data[col_idx]
                flattened = img.flatten()
                val = int(flattened[row_idx])
                # Use MathTex for monospace alignment
                val_text = MathTex(f"{val:3d}", font_size=24, color=WHITE)
                val_text.move_to([start_x + col_idx * col_spacing, 0, 0])
                row_values.append(val_text)
            row_group = VGroup(*row_values)
            matrix_entries.append(row_group)

        # Add vertical dots to indicate many more rows
        dots_values = []
        for col_idx in range(16):
            dot = Text("⋮", font_size=12, color=WHITE)
            dot.move_to([start_x + col_idx * col_spacing, 0, 0])
            dots_values.append(dot)
        dots_row = VGroup(*dots_values)
        matrix_entries.append(dots_row)

        # Show rows near end but still with digit data (rows 480-487, before bottom padding)
        for row_idx in range(480, 488):
            row_values = []
            for col_idx in range(16):
                img = all_image_data[col_idx]
                flattened = img.flatten()
                val = int(flattened[row_idx])
                # Use MathTex for monospace alignment
                val_text = MathTex(f"{val:3d}", font_size=24, color=WHITE)
                val_text.move_to([start_x + col_idx * col_spacing, 0, 0])
                row_values.append(val_text)
            row_group = VGroup(*row_values)
            matrix_entries.append(row_group)

        # Arrange all rows with more spacing
        data_matrix_content = VGroup(*matrix_entries).arrange(DOWN, buff=0.08)

        # Create brackets manually using Lines (LaTeX vphantom doesn't render properly)
        matrix_height = data_matrix_content.get_height()
        matrix_top = data_matrix_content.get_top()[1]
        matrix_bottom = data_matrix_content.get_bottom()[1]
        matrix_left = data_matrix_content.get_left()[0]
        matrix_right = data_matrix_content.get_right()[0]

        bracket_width = 0.12
        bracket_gap = 0.15
        stroke_w = 3

        # Left bracket "[" shape
        left_bracket = VGroup(
            Line(start=[matrix_left - bracket_gap, matrix_top, 0],
                 end=[matrix_left - bracket_gap - bracket_width, matrix_top, 0], color=WHITE, stroke_width=stroke_w),
            Line(start=[matrix_left - bracket_gap - bracket_width, matrix_top, 0],
                 end=[matrix_left - bracket_gap - bracket_width, matrix_bottom, 0], color=WHITE, stroke_width=stroke_w),
            Line(start=[matrix_left - bracket_gap - bracket_width, matrix_bottom, 0],
                 end=[matrix_left - bracket_gap, matrix_bottom, 0], color=WHITE, stroke_width=stroke_w),
        )

        # Right bracket "]" shape
        right_bracket = VGroup(
            Line(start=[matrix_right + bracket_gap, matrix_top, 0],
                 end=[matrix_right + bracket_gap + bracket_width, matrix_top, 0], color=WHITE, stroke_width=stroke_w),
            Line(start=[matrix_right + bracket_gap + bracket_width, matrix_top, 0],
                 end=[matrix_right + bracket_gap + bracket_width, matrix_bottom, 0], color=WHITE, stroke_width=stroke_w),
            Line(start=[matrix_right + bracket_gap + bracket_width, matrix_bottom, 0],
                 end=[matrix_right + bracket_gap, matrix_bottom, 0], color=WHITE, stroke_width=stroke_w),
        )

        data_matrix_full = VGroup(left_bracket, data_matrix_content, right_bracket)
        data_matrix_full.move_to(DOWN * 0.5)  # Move down to avoid overlap

        # Add label showing dimensions (scaled up)
        matrix_label = Text("", font_size=24, color=DARK_GREEN, weight=BOLD)  # Increased from 20 to 24
        matrix_label.next_to(data_matrix_full, DOWN, buff=0.4)  # Increased spacing

        # Fade out individual vectors and show the combined matrix
        self.play(*[FadeOut(all_vectors[i]) for i in range(16)], run_time=0.8)
        self.add_fixed_in_frame_mobjects(data_matrix_full, matrix_label)
        self.play(FadeIn(data_matrix_full), FadeIn(matrix_label), run_time=1.5)
        self.wait(5.22)  # Extended for narration segment 4

        # Fade out matrix for next step
        self.play(FadeOut(data_matrix_full), FadeOut(matrix_label), run_time=0.8)

        # ====================================================================
        # STEP 5: Show Clustered Result with Real MNIST Images
        # ====================================================================

        step_title5 = Text("Step 5: 2D Clustered Result", font_size=26, color=DARK_ORANGE, weight=BOLD)
        step_title5.move_to(UP * 2.5)

        self.add_fixed_in_frame_mobjects(step_title5)
        self.play(FadeOut(step_title_matrix), FadeIn(step_title5))
        
        # Create clustered visualization with real MNIST images (WHITE color)
        # Use manually positioned clusters that look like natural t-SNE results
        clustered_images = VGroup()

        # Define cluster positions for 8 digits (2 samples each)
        # Organic scattered positioning like real t-SNE output
        # Clusters are NOT in a grid - some overlap, irregular spacing
        cluster_positions = {
            0: [[-3.8, 1.2], [-4.3, 0.6]],      # Upper left area
            1: [[4.5, 1.0], [4.2, 0.4]],        # Upper right area
            2: [[-0.5, -2.8], [-1.0, -2.3]],    # Lower center
            3: [[-2.2, -0.4], [-1.7, -1.0]],    # Center-left (between clusters)
            4: [[2.8, -1.8], [3.3, -2.4]],      # Lower right
            5: [[0.3, 1.3], [0.8, 0.7]],        # Upper center
            6: [[-5.0, -1.5], [-4.5, -2.0]],    # Left side, lower
            7: [[1.5, -0.2], [2.1, 0.3]]        # Center-right area
        }

        # Calculate cluster centers for labels
        cluster_centers = {}
        for digit in range(8):
            positions = cluster_positions[digit]
            center_x = sum(p[0] for p in positions) / len(positions)
            center_y = max(p[1] for p in positions)  # Get topmost y position
            cluster_centers[digit] = [center_x, center_y]

        for idx in range(16):
            img = all_image_data[idx]
            label = selected_labels[idx]

            # Get position from predefined clusters
            digit_count = sum(1 for i in range(idx) if selected_labels[i] == label)
            x, y = cluster_positions[label][digit_count]

            # Create real MNIST visualization at cluster position (WHITE color)
            mnist_viz = create_mnist_visual(img, WHITE, scale=0.05)
            mnist_viz.move_to(np.array([x, y, 0]))
            clustered_images.add(mnist_viz)

        # Show clustered real images
        self.play(*[FadeIn(img, scale=0.5) for img in clustered_images], run_time=2.5)
        self.wait(1)

        # Add cluster labels positioned above each cluster (WHITE color)
        cluster_labels = VGroup()
        for digit in range(8):
            center_x, center_y = cluster_centers[digit]
            label_text = Text(str(digit), font_size=28, color=WHITE, weight=BOLD)
            label_text.move_to(np.array([center_x, center_y + 0.5, 0]))
            cluster_labels.add(label_text)

        self.add_fixed_in_frame_mobjects(cluster_labels)
        self.play(*[FadeIn(l, scale=1.3) for l in cluster_labels], run_time=1)
        self.wait(2)
        
        # Message at bottom
        message = Text(
            "✓ Same digits cluster together!",
            font_size=22,
            color=DARK_GREEN,
            weight=BOLD
        )
        message.move_to(DOWN * 3.5)  # Moved further down to avoid cluster overlap
        constrain_width(message, 0.8)
        
        self.add_fixed_in_frame_mobjects(message)
        self.play(FadeIn(message))
        self.wait(2.5)
        
        # ====================================================================
        # Comparison: Why t-SNE?
        # ====================================================================

        # Clear all including the main title
        self.play(*[FadeOut(m) for m in [step_title5, clustered_images, cluster_labels, message, title]])

        comp_title = Text("Why t-SNE?", font_size=64, color=TECH_CYAN, weight=BOLD)
        comp_title.move_to(UP * 3.3)

        self.add_fixed_in_frame_mobjects(comp_title)
        self.play(Write(comp_title))
        self.wait(1.19)  # Extended for narration segment 6
        
        # Generate comparison data - MORE POINTS (lots of points as user requested)
        np.random.seed(123)
        cluster_data = []
        for cluster_idx, center in enumerate([np.array([-1.5, 1.5]), np.array([1.5, 1.5]), np.array([0, -1.5])]):
            for _ in range(50):  # Increased from 10 to 50 for "lots of points"
                point = center + np.random.randn(2) * 0.4
                cluster_data.append(point)

        # List of methods to show
        methods = ["PCA", "Isomap", "Sammon Map", "LLE"]

        for method_name in methods:
            method_label = Text(method_name, font_size=28, color=WHITE, weight=BOLD)
            method_label.move_to(UP * 2.0)
            self.add_fixed_in_frame_mobjects(method_label)
            self.play(FadeIn(method_label))

            # Generate points for this method (overlapping/poor clustering)
            method_points = VGroup()
            for i, data in enumerate(cluster_data):
                y_pos = data[1] * 0.2 + np.random.randn() * 0.3
                y_pos = np.clip(y_pos, -2.0, 2.0)
                pos = np.array([data[0] * 0.3 + np.random.randn() * 0.25, y_pos, 0])
                method_points.add(Dot(pos, color=WHITE, radius=0.05))

            self.play(*[GrowFromCenter(d) for d in method_points], run_time=1.5)
            self.wait(0.8)
            self.play(FadeOut(method_label), FadeOut(method_points), run_time=0.6)

        # Show "However the performance is not good" message
        performance_message = Text(
            "However the performance is not good",
            font_size=36,
            color=DARK_RED,
            weight=BOLD
        )
        performance_message.move_to(ORIGIN)
        constrain_width(performance_message, 0.85)

        self.add_fixed_in_frame_mobjects(performance_message)
        self.play(FadeIn(performance_message))
        self.wait(2.0)
        self.play(FadeOut(performance_message))

        # t-SNE
        method2 = Text("t-SNE", font_size=28, color=WHITE, weight=BOLD)
        method2.move_to(UP * 2.0)
        self.add_fixed_in_frame_mobjects(method2)
        self.play(FadeIn(method2))

        tsne_points = VGroup()
        tsne_centers = [LEFT * 1.9, RIGHT * 1.9, DOWN * 1.3]
        for i, data in enumerate(cluster_data):
            cluster_idx = i // 50  # Updated to match new cluster size
            offset = np.random.randn(2) * 0.18
            pos = tsne_centers[cluster_idx] + np.array([offset[0], offset[1], 0])
            tsne_points.add(Dot(pos, color=WHITE, radius=0.05))

        self.play(*[GrowFromCenter(d) for d in tsne_points], run_time=1.5)
        self.wait(1.0)

        # Author credits
        credits = Text(
            "L. van der Maaten & G. Hinton (2008)",
            font_size=22,
            color=TEXT_COLOR
        )
        credits.move_to(DOWN * 2.5)
        constrain_width(credits, 0.75)

        self.add_fixed_in_frame_mobjects(credits)
        self.play(FadeIn(credits))
        self.wait(2)

        # Clear all
        self.play(*[FadeOut(m) for m in [comp_title, method2, tsne_points, credits]])
        self.wait(0.5)
    
    def scene_03_tsne_math(self):
        # Title
        title = Text(
            "t-SNE: Mathematical Formulation",
            font_size=44,
            color=DARK_BLUE,
            weight=BOLD
        ).to_edge(UP)
        
        self.play(Write(title))
        self.wait(1)
        
        # High-dimensional similarities
        hd_title = Text(
            "High-Dimensional Similarities (P):",
            font_size=28,
            color=DARK_ORANGE,
            weight=BOLD
        ).next_to(title, DOWN, buff=0.7).to_edge(LEFT, buff=0.5)
        
        hd_formula = MathTex(
            r"p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}",
            color=TEXT_COLOR,
            font_size=28
        ).next_to(hd_title, DOWN, buff=0.3)
        
        # Ensure equation fits within frame
        if hd_formula.get_width() > config.frame_width * 0.9:
            hd_formula.set_width(config.frame_width * 0.9)
        
        hd_symmetric = MathTex(
            r"p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}",
            color=TEXT_COLOR,
            font_size=28
        ).next_to(hd_formula, DOWN, buff=0.3)
        
        symmetric_note = Text(
            "(symmetrized)",
            font_size=20,
            color=LIGHT_GRAY,
            slant=ITALIC
        ).next_to(hd_symmetric, RIGHT, buff=0.3)
        
        self.play(Write(hd_title))
        self.wait(0.5)
        self.play(Write(hd_formula), run_time=2)
        self.wait(1)
        self.play(Write(hd_symmetric), FadeIn(symmetric_note))
        self.wait(2)
        
        # Low-dimensional similarities
        ld_title = Text(
            "Low-Dimensional Similarities (Q):",
            font_size=28,
            color=DARK_GREEN,
            weight=BOLD
        ).next_to(hd_symmetric, DOWN, buff=0.7).to_edge(LEFT, buff=0.5)
        
        ld_formula = MathTex(
            r"q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{a \neq b} (1 + \|y_a - y_b\|^2)^{-1}}",
            color=TEXT_COLOR,
            font_size=28
        ).next_to(ld_title, DOWN, buff=0.3)
        
        # Ensure equation fits within frame
        if ld_formula.get_width() > config.frame_width * 0.9:
            ld_formula.set_width(config.frame_width * 0.9)
        
        cauchy_note = Text(
            "(Cauchy/Student-t kernel)",
            font_size=20,
            color=LIGHT_GRAY,
            slant=ITALIC
        ).next_to(ld_formula, DOWN, buff=0.2)
        
        self.play(Write(ld_title))
        self.wait(0.5)
        self.play(Write(ld_formula), run_time=2)
        self.play(FadeIn(cauchy_note))
        self.wait(2)
        
        # Fade out previous content to make room for cost function
        self.play(
            FadeOut(hd_title),
            FadeOut(hd_formula),
            FadeOut(hd_symmetric),
            FadeOut(symmetric_note),
            FadeOut(ld_title),
            FadeOut(ld_formula),
            FadeOut(cauchy_note),
            run_time=1
        )
        
        # Cost function - now positioned from title
        cost_title = Text(
            "Cost Function (KL Divergence):",
            font_size=32,
            color=DARK_RED,
            weight=BOLD
        ).next_to(title, DOWN, buff=1.2)
        
        cost_formula = MathTex(
            r"C(Y) = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}",
            color=TEXT_COLOR,
            font_size=36
        ).next_to(cost_title, DOWN, buff=0.5)
        
        # Highlight box
        cost_box = SurroundingRectangle(
            cost_formula,
            color=DARK_RED,
            buff=0.3,
            stroke_width=3
        )
        
        self.play(Write(cost_title))
        self.wait(0.5)
        self.play(Write(cost_formula), run_time=2)
        self.play(Create(cost_box))
        self.wait(7.93)  # Extended for narration segment 13
        
        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects])
    
    # ========================================================================
    # SCENE 4: k-DISTANCE DEFINITION
    # ========================================================================
    
    def scene_04_kdistance(self):
        # ====================================================================
        # MOTIVATION SECTION
        # ====================================================================

        motivation_title = Text(
            "Motivation",
            font_size=48,
            color=DARK_BLUE,
            weight=BOLD
        ).to_edge(UP)

        self.play(Write(motivation_title))
        self.wait(1)

        # First motivation text
        motivation_text1 = Text(
            "We are curious about whether the geometric properties\nof high-dimensional points will be preserved\nto the low-dimensional points",
            font_size=30,
            color=TEXT_COLOR
        ).move_to(ORIGIN)
        constrain_width(motivation_text1, 0.85)

        self.play(FadeIn(motivation_text1))
        self.wait(6.89)  # Extended for narration segment 14
        self.play(FadeOut(motivation_text1))

        # Second motivation text
        motivation_text2 = Text(
            "So we want to study k-distance",
            font_size=36,
            color=DARK_GREEN,
            weight=BOLD
        ).move_to(ORIGIN)
        constrain_width(motivation_text2, 0.85)

        self.play(FadeIn(motivation_text2))
        self.wait(2.5)
        self.play(FadeOut(motivation_text2), FadeOut(motivation_title))
        self.wait(0.5)

        # ====================================================================
        # WHAT IS K-DISTANCE
        # ====================================================================

        # Title
        title = Text(
            "What is k-distance?",
            font_size=48,
            color=DARK_BLUE,
            weight=BOLD
        ).to_edge(UP)

        self.play(Write(title))
        self.wait(1)
        
        # Definition
        definition = VGroup(
            Text(
                "A finite set S in d-dimensional space where",
                font_size=28,
                color=TEXT_COLOR
            ),
            Text(
                "all pairwise distances take exactly k distinct values",
                font_size=28,
                color=TEXT_COLOR
            ),
            Text(
                "{Δ₁, Δ₂, ..., Δₖ}",
                font_size=28,
                color=TEXT_COLOR
            )
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(title, DOWN, buff=0.8)
        
        def_box = SurroundingRectangle(definition, color=SECONDARY_COLOR, buff=0.3)
        
        self.play(Write(definition), run_time=2.5)
        self.play(Create(def_box))
        self.wait(6.05)  # Extended for narration segment 16
        
        self.play(FadeOut(definition), FadeOut(def_box))
        self.wait(0.5)

        # Clear the "What is k-distance?" title
        self.play(FadeOut(title))
        self.wait(0.3)

        # ===================================================================
        # 1-DISTANCE (equilateral triangle) - BIG TITLE
        # ===================================================================
        ex1_title = Text(
            "1-distance",
            font_size=48,
            color=DARK_ORANGE,
            weight=BOLD
        ).to_edge(UP)

        # Ensure title fits
        constrain_width(ex1_title, 0.9)
        
        # Create equilateral triangle - centered
        triangle = RegularPolygon(
            n=3,
            color=DARK_GREEN,
            stroke_width=4
        ).scale(2.0).move_to(ORIGIN)

        triangle_dots = VGroup(*[
            Dot(triangle.get_vertices()[i], color=DARK_BLUE, radius=0.15)
            for i in range(3)
        ])

        # Distance labels (using d instead of Delta_1)
        dist_labels_1 = VGroup(*[
            MathTex(r"d", color=DARK_GREEN, font_size=32).move_to(
                (triangle.get_vertices()[i] + triangle.get_vertices()[(i+1)%3]) / 2
            ).shift(
                (triangle.get_vertices()[i] + triangle.get_vertices()[(i+1)%3]) / 2 * 0.25
            )
            for i in range(3)
        ])

        eq_label = Text(
            "All distances equal",
            font_size=28,
            color=TEXT_COLOR
        ).next_to(triangle, DOWN, buff=0.8)

        self.play(Write(ex1_title))
        self.wait(0.5)
        self.play(Create(triangle), *[GrowFromCenter(d) for d in triangle_dots])
        self.play(Write(dist_labels_1))
        self.play(FadeIn(eq_label))
        self.wait(3.54)  # Extended for narration segment 17

        # Clear 1-distance visualization
        self.play(
            FadeOut(ex1_title),
            FadeOut(triangle),
            FadeOut(triangle_dots),
            FadeOut(dist_labels_1),
            FadeOut(eq_label)
        )
        self.wait(0.3)

        # ===================================================================
        # THEOREM 1 (original Theorem 2.1) - BIG TITLE
        # ===================================================================
        theorem1_title = Text(
            "Theorem 1",
            font_size=48,
            color=DARK_ORANGE,
            weight=BOLD
        ).to_edge(UP)

        theorem1_text = VGroup(
            Text(
                "For a bounded 1-distance set {xᵢ ∈ ℝᵖ},",
                font_size=30,
                color=TEXT_COLOR
            ),
            Text(
                "any visualization points {yᵢ ∈ ℝ²}",
                font_size=30,
                color=TEXT_COLOR
            ),
            Text(
                "is also a 1-distance set",
                font_size=30,
                color=TEXT_COLOR
            )
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(ORIGIN)

        # Constrain width
        for line in theorem1_text:
            constrain_width(line, 0.85)

        self.play(Write(theorem1_title))
        self.wait(0.5)

        for line in theorem1_text:
            self.play(Write(line), run_time=1.2)
            self.wait(0.5)

        self.wait(4.47)  # Extended for narration segment 18

        # Clear theorem
        self.play(FadeOut(theorem1_title), *[FadeOut(line) for line in theorem1_text])
        self.wait(0.3)

        # ===================================================================
        # k-DISTANCE (k≥2) - BIG TITLE
        # ===================================================================
        k_dist_title = Text(
            "k-distance (k ≥ 2)",
            font_size=48,
            color=DARK_PURPLE,
            weight=BOLD
        ).to_edge(UP)

        # First part: "The minimum for C exists, but for more detail,"
        k_dist_text = VGroup(
            Text(
                "The minimum for C exists,",
                font_size=36,
                color=TEXT_COLOR
            ),
            Text(
                "but for more detail,",
                font_size=36,
                color=TEXT_COLOR
            )
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(ORIGIN)

        # Constrain width
        for line in k_dist_text:
            constrain_width(line, 0.85)

        self.play(Write(k_dist_title))
        self.wait(0.5)

        for line in k_dist_text:
            self.play(FadeIn(line), run_time=1.2)
            self.wait(0.8)

        self.wait(2)

        # Vanish all words (title and text)
        self.play(FadeOut(k_dist_title), *[FadeOut(line) for line in k_dist_text], run_time=1)
        self.wait(0.5)

        # Show BIG final text
        big_final_text = Text(
            "come to 科教館 (2/4)!",
            font_size=72,
            color=DARK_GREEN,
            weight=BOLD
        ).move_to(ORIGIN)
        constrain_width(big_final_text, 0.9)

        self.play(FadeIn(big_final_text, scale=1.2), run_time=1.5)
        self.wait(4)

        # Clear
        self.play(FadeOut(big_final_text), run_time=1)


# ============================================================================
# RENDERING INSTRUCTIONS
# ============================================================================

"""
To render the complete presentation:
    manim -pql tsne_kdistance_full.py TSNEPresentation
    
Quality options:
    -ql : Low quality (480p) - fast rendering for testing
    -qm : Medium quality (720p)
    -qh : High quality (1080p) - recommended for final video
    -qk : 4K quality (2160p) - best quality but slow rendering

Other useful flags:
    -p  : Preview after rendering
    --format=gif : Export as GIF instead of video
    --save_sections : Save individual scenes as separate files
    
Example commands:
    manim -pqh tsne_kdistance_full.py TSNEPresentation
    manim -pql --format=gif tsne_kdistance_full.py TSNEPresentation
"""