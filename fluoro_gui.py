"""
FluoroQuant - GUI Module
Handles all user interface components
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os

class FluoroGUI:
    """GUI handler for FluoroQuant"""
    
    def __init__(self, app):
        self.app = app
        self.main_window = None
        
        # GUI variables
        self.channel_status_vars = {}
        self.channel_enable_vars = {}
        self.processing_mode = None
        self.preview_var = None
        self.auto_update_var = None
        self.auto_export_batch = None
        
        # Control variables
        self.mode_var = None
        self.threshold_method = None
        self.manual_thresh_var = None
        self.parameter_vars = {}
        
        # Export variables
        self.export_csv = None
        self.export_json = None
        self.export_excel = None
        self.export_images = None
        self.export_overlays = None
        
        # Display elements
        self.fig = None
        self.axes = None
        self.canvas = None
        self.stats_text = None
        self.progress_bar = None
        self.progress_var = None
        self.update_timer = None
        self.update_pending = False
        
        # Color scheme
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#00ff88',
            'secondary': '#ff6b35',
            'panel': '#3a3a3a',
            'button_bg': '#4a4a4a',
            'entry_bg': '#525252',
            'text_bg': '#404040',
            'ch1_color': '#ff4444',
            'ch2_color': '#44ff44',
            'ch3_color': '#4444ff'
        }

        # Channel names mapping
        self.channel_names = {'ch1': 'Channel 1', 'ch2': 'Channel 2', 'ch3': 'Channel 3'}
    
    def create_main_window(self):
        """Create the main application window"""
        self.main_window = tk.Tk()
        self.main_window.title("FluoroQuant v2.0 - Multi-Channel Analysis")
        self.main_window.geometry("1600x900")
        self.main_window.configure(bg=self.colors['bg'])
        
        # Configure dark theme
        self.configure_dark_theme()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main layout
        self.create_layout()
        
        # Initialize display
        self.update_preview_display(self.app)
    
    def configure_dark_theme(self):
        """Configure ttk widgets for dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure all widget styles
        style.configure('TLabel', 
                       background=self.colors['bg'], 
                       foreground=self.colors['fg'])
        
        style.configure('TButton',
                       background=self.colors['button_bg'],
                       foreground=self.colors['fg'],
                       borderwidth=1,
                       relief='flat',
                       focuscolor='none')
        style.map('TButton',
                 background=[('active', self.colors['accent']),
                            ('pressed', self.colors['secondary']),
                            ('disabled', self.colors['panel'])],
                 foreground=[('active', '#000000'),
                            ('disabled', '#666666')])
        
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('TLabelframe', 
                       background=self.colors['bg'], 
                       foreground=self.colors['fg'],
                       bordercolor=self.colors['panel'])
        style.configure('TLabelframe.Label', 
                       background=self.colors['bg'], 
                       foreground=self.colors['accent'])
        
        style.configure('TNotebook', 
                       background=self.colors['panel'],
                       borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=self.colors['panel'],
                       foreground=self.colors['fg'],
                       padding=[20, 10])
        style.map('TNotebook.Tab',
                 background=[('selected', self.colors['button_bg'])],
                 foreground=[('selected', self.colors['accent'])])
        
        style.configure('TCheckbutton',
                       background=self.colors['bg'],
                       foreground=self.colors['fg'],
                       focuscolor='none')
        
        style.configure('TRadiobutton',
                       background=self.colors['bg'],
                       foreground=self.colors['fg'],
                       focuscolor='none')
        
        style.configure('TCombobox',
                       fieldbackground=self.colors['entry_bg'],
                       background=self.colors['button_bg'],
                       foreground=self.colors['fg'],
                       arrowcolor=self.colors['fg'],
                       bordercolor=self.colors['panel'])
        
        style.configure('TScale',
                       background=self.colors['bg'],
                       troughcolor=self.colors['panel'],
                       bordercolor=self.colors['panel'])
        
        style.configure('TProgressbar',
                       background=self.colors['accent'],
                       troughcolor=self.colors['panel'],
                       bordercolor=self.colors['panel'])
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.main_window, 
                         bg=self.colors['panel'], 
                         fg=self.colors['fg'],
                         activebackground=self.colors['accent'],
                         activeforeground='#000000')
        self.main_window.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0,
                           bg=self.colors['panel'],
                           fg=self.colors['fg'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Channel 1", 
                             command=lambda: self.load_channel_dialog('ch1'))
        file_menu.add_command(label="Load Channel 2", 
                             command=lambda: self.load_channel_dialog('ch2'))
        file_menu.add_command(label="Load Channel 3", 
                             command=lambda: self.load_channel_dialog('ch3'))
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", 
                             command=lambda: self.app.export_results('current'))
        file_menu.add_separator()
        file_menu.add_command(label="Reset All", command=self.app.reset_all)
        file_menu.add_command(label="Exit", command=self.main_window.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0,
                               bg=self.colors['panel'],
                               fg=self.colors['fg'])
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Process Current", 
                                 command=self.app.process_current_image)
        analysis_menu.add_command(label="Batch Process", 
                                 command=self.app.start_batch_processing)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0,
                           bg=self.colors['panel'],
                           fg=self.colors['fg'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_guide)
    
    def create_layout(self):
        """Create main window layout"""
        # Main container
        main_container = ttk.Frame(self.main_window)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create paned window for resizable layout
        paned = ttk.PanedWindow(main_container, orient='horizontal')
        paned.pack(fill='both', expand=True)
        
        # Left panel - Channels and Preview
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=3)
        
        self.create_channel_section(left_panel)
        self.create_preview_section(left_panel)
        
        # Right panel - Controls and Results
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=1)
        
        self.create_control_tabs(right_panel)
        self.create_results_section(right_panel)
        
        # Bottom panel - Progress
        self.create_progress_section(main_container)
    
    def create_channel_section(self, parent):
        """Create channel loading section"""
        # Title
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(title_frame, 
                               text="🔬 FluoroQuant", 
                               font=('Arial', 20, 'bold'))
        title_label.pack(side='left')
        
        version_label = ttk.Label(title_frame, 
                                 text="v2.0", 
                                 font=('Arial', 10),
                                 foreground='#888888')
        version_label.pack(side='left', padx=(10, 0))
        
        # Channel frame
        channel_frame = ttk.LabelFrame(parent, text="Channel Management", padding=10)
        channel_frame.pack(fill='x', pady=(0, 10))
        
        # Channel grid
        channel_grid = ttk.Frame(channel_frame)
        channel_grid.pack(fill='x')
        
        for i, ch in enumerate(['ch1', 'ch2', 'ch3']):
            self.create_channel_controls(channel_grid, ch, i)
        
        # Mode selection
        mode_frame = ttk.Frame(channel_frame)
        mode_frame.pack(fill='x', pady=(15, 0))
        
        # Processing mode
        ttk.Label(mode_frame, text="Mode:").pack(side='left')
        self.processing_mode = tk.StringVar(value="single")
        
        single_btn = ttk.Radiobutton(mode_frame, 
                                    text="Single Image",
                                    variable=self.processing_mode,
                                    value="single",
                                    command=self.on_mode_change)
        single_btn.pack(side='left', padx=(10, 0))
        
        batch_btn = ttk.Radiobutton(mode_frame,
                                   text="Batch Processing",
                                   variable=self.processing_mode,
                                   value="batch",
                                   command=self.on_mode_change)
        batch_btn.pack(side='left', padx=(10, 0))
        
        self.batch_frame = ttk.Frame(channel_frame)
        self.batch_button = ttk.Button(self.batch_frame,
                                      text="📁 Select Folder",
                                      command=self.select_batch_folder,
                                      state='disabled')
        self.batch_button.pack(side='left')
        
        self.batch_status = tk.StringVar(value="")
        ttk.Label(self.batch_frame, 
                 textvariable=self.batch_status,
                 foreground='#cccccc').pack(side='left', padx=(10, 0))
        
        # Pattern display
        self.pattern_frame = ttk.Frame(channel_frame)
        self.pattern_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(self.pattern_frame, text="Pattern:").pack(side='left')
        self.pattern_var = tk.StringVar(value="Load 2+ channels to learn pattern")
        ttk.Label(self.pattern_frame,
                 textvariable=self.pattern_var,
                 foreground=self.colors['accent']).pack(side='left', padx=(5, 0))
    
    def create_channel_controls(self, parent, channel, column):
        """Create controls for a single channel"""
        ch_frame = ttk.Frame(parent)
        ch_frame.grid(row=0, column=column, padx=10, sticky='ew')
        parent.columnconfigure(column, weight=1)
        
        # Header with color
        header_frame = ttk.Frame(ch_frame)
        header_frame.pack(fill='x')
        
        # Color indicator
        color_label = tk.Label(header_frame,
                              text="●",
                              font=('Arial', 20),
                              bg=self.colors['bg'],
                              fg=self.colors[f'{channel}_color'])
        color_label.pack(side='left')
        
        # Channel name
        ch_name = {'ch1': 'Channel 1', 'ch2': 'Channel 2', 'ch3': 'Channel 3'}[channel]
        ttk.Label(header_frame, 
                 text=ch_name,
                 font=('Arial', 11, 'bold')).pack(side='left', padx=(5, 0))
        
        # Load button
        load_btn = ttk.Button(ch_frame,
                             text="Load Image",
                             command=lambda: self.load_channel_dialog(channel))
        load_btn.pack(fill='x', pady=(5, 2))
        
        # Status
        status_var = tk.StringVar(value="No file loaded")
        self.channel_status_vars[channel] = status_var
        status_label = ttk.Label(ch_frame,
                                textvariable=status_var,
                                font=('Arial', 9),
                                foreground='#cccccc')
        status_label.pack(fill='x')
        
        # Enable checkbox
        enable_var = tk.BooleanVar(value=False)
        self.channel_enable_vars[channel] = enable_var
        enable_check = ttk.Checkbutton(ch_frame,
                                      text="Active",
                                      variable=enable_var,
                                      command=lambda: self.toggle_channel(channel))
        enable_check.pack(pady=(5, 0))

    def unified_process_command(self):
        """Unified command that processes single image or batch based on current mode"""
        try:
            print(f"Unified process command called. Mode: {self.processing_mode.get()}")
            
            if self.processing_mode.get() == 'batch':
                # Check if batch is ready
                if not hasattr(self.app, 'detected_groups') or not self.app.detected_groups:
                    self.show_warning("No image groups detected. Select a batch folder first.")
                    return
                
                print(f"Starting batch processing with {len(self.app.detected_groups)} groups")
                # Run batch processing
                self.app.start_batch_processing()
            else:
                # Check for active channels
                active_channels = [ch for ch, active in self.app.active_channels.items() if active]
                if not active_channels:
                    self.show_warning("No active channels selected. Load and activate channels first.")
                    return
                    
                print(f"Processing single image with channels: {active_channels}")
                # Run single image processing
                self.app.process_current_image()
                
        except Exception as e:
            print(f"Unified process command error: {e}")
            self.show_error(f"Processing error: {str(e)}")

    def create_preview_section(self, parent):
        """Create image preview section"""
        preview_frame = ttk.LabelFrame(parent, text="Preview & Analysis", padding=10)
        preview_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Controls
        control_frame = ttk.Frame(preview_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # View selector
        ttk.Label(control_frame, text="View:").pack(side='left')
        self.preview_var = tk.StringVar(value='ch1')
        preview_combo = ttk.Combobox(control_frame,
                                    textvariable=self.preview_var,
                                    values=['ch1', 'ch2', 'ch3', 'composite'],
                                    state='readonly',
                                    width=12)
        preview_combo.pack(side='left', padx=(5, 15))
        preview_combo.bind('<<ComboboxSelected>>', 
                        lambda e: self.update_preview_display(self.app))
        
        self.process_btn = ttk.Button(control_frame,
                                    text="⚡ Process",
                                    command=self.unified_process_command,
                                    state='disabled')
        self.process_btn.pack(side='left', padx=(0, 10))
        
        # Auto-update checkbox
        self.auto_update_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame,
                    text="Auto-update",
                    variable=self.auto_update_var).pack(side='left')
        
        self.fig = plt.figure(figsize=(12, 7), facecolor=self.colors['panel'])
        self.axes = self.fig.subplots(2, 3)
        
        for ax_row in self.axes:
            for ax in ax_row:
                ax.set_facecolor(self.colors['bg'])
                ax.tick_params(colors=self.colors['fg'], labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor(self.colors['panel'])
                    spine.set_linewidth(0.5)
                ax.grid(False)
        
        self.fig.tight_layout(pad=2.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, preview_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    
    def create_control_tabs(self, parent):
        """Create tabbed control panel"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True, pady=(0, 10))
        
        # Quick controls
        quick_frame = ttk.Frame(notebook)
        notebook.add(quick_frame, text="Quick")
        self.create_quick_controls(quick_frame)
        
        # Advanced controls
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        self.create_advanced_controls(advanced_frame)
        
        # Multi-channel
        multi_frame = ttk.Frame(notebook)
        notebook.add(multi_frame, text="Multi-Channel")
        self.create_multichannel_controls(multi_frame)
        
        # Export
        export_frame = ttk.Frame(notebook)
        notebook.add(export_frame, text="Export")
        self.create_export_controls(export_frame)
    
    def create_quick_controls(self, parent):
        """Create quick control panel"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Analysis mode
        mode_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Mode", padding=10)
        mode_frame.pack(fill='x', padx=5, pady=5)
        
        self.mode_var = tk.StringVar(value='general')
        modes = [('General', 'general'), ('Neurons', 'neurons'), 
                ('Cells', 'cells'), ('Colonies', 'colonies')]
        
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var,
                           value=value).pack(anchor='w')
        
        # Threshold method
        thresh_frame = ttk.LabelFrame(scrollable_frame, text="Threshold", padding=10)
        thresh_frame.pack(fill='x', padx=5, pady=5)
        
        self.threshold_method = tk.StringVar(value='otsu')
        methods = [('Otsu', 'otsu'), ('Triangle', 'triangle'), 
                  ('Li', 'li'), ('Manual', 'manual')]
        
        for text, value in methods:
            ttk.Radiobutton(thresh_frame, text=text, 
                           variable=self.threshold_method,
                           value=value,
                           command=self.on_parameter_change).pack(anchor='w')
        
        # Manual threshold
        self.manual_thresh_var = tk.IntVar(value=128)
        self.create_slider(thresh_frame, "Manual:", self.manual_thresh_var, 0, 255, 128)
        
        # Basic preprocessing
        preproc_frame = ttk.LabelFrame(scrollable_frame, text="Preprocessing", padding=10)
        preproc_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['gaussian'] = tk.BooleanVar()
        ttk.Checkbutton(preproc_frame, text="Smooth", 
                       variable=self.parameter_vars['gaussian'],
                       command=self.on_parameter_change).pack(anchor='w')
        
        self.parameter_vars['median'] = tk.BooleanVar()
        ttk.Checkbutton(preproc_frame, text="Denoise", 
                       variable=self.parameter_vars['median'],
                       command=self.on_parameter_change).pack(anchor='w')
        
        self.parameter_vars['clahe'] = tk.BooleanVar()
        ttk.Checkbutton(preproc_frame, text="Enhance", 
                       variable=self.parameter_vars['clahe'],
                       command=self.on_parameter_change).pack(anchor='w')
        
        self.parameter_vars['morphology'] = tk.BooleanVar()
        ttk.Checkbutton(preproc_frame, text="Clean", 
                       variable=self.parameter_vars['morphology'],
                       command=self.on_parameter_change).pack(anchor='w')
        
        # Size filter
        size_frame = ttk.LabelFrame(scrollable_frame, text="Size Filter", padding=10)
        size_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['min_size'] = tk.IntVar(value=50)
        self.create_slider(size_frame, "Min:", self.parameter_vars['min_size'], 1, 500, 50)
        
        self.parameter_vars['max_size'] = tk.IntVar(value=10000)
        self.create_slider(size_frame, "Max:", self.parameter_vars['max_size'], 100, 50000, 10000)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_advanced_controls(self, parent):
        """Create advanced control panel"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Preprocessing parameters
        prep_frame = ttk.LabelFrame(scrollable_frame, text="Preprocessing Parameters", padding=10)
        prep_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['gaussian_sigma'] = tk.DoubleVar(value=1.0)
        self.create_slider(prep_frame, "Gaussian σ:", 
                          self.parameter_vars['gaussian_sigma'], 0.1, 5.0, 1.0)
        
        self.parameter_vars['median_size'] = tk.IntVar(value=3)
        self.create_slider(prep_frame, "Median Size:", 
                          self.parameter_vars['median_size'], 1, 15, 3)
        
        self.parameter_vars['gamma'] = tk.DoubleVar(value=1.0)
        self.create_slider(prep_frame, "Gamma:", 
                          self.parameter_vars['gamma'], 0.1, 3.0, 1.0)
        
        self.parameter_vars['contrast'] = tk.DoubleVar(value=1.0)
        self.create_slider(prep_frame, "Contrast:", 
                          self.parameter_vars['contrast'], 0.1, 3.0, 1.0)
        
        self.parameter_vars['brightness'] = tk.IntVar(value=0)
        self.create_slider(prep_frame, "Brightness:", 
                          self.parameter_vars['brightness'], -100, 100, 0)
        
        self.parameter_vars['clahe_clip'] = tk.DoubleVar(value=2.0)
        self.create_slider(prep_frame, "CLAHE Clip:", 
                          self.parameter_vars['clahe_clip'], 1.0, 10.0, 2.0)
        
        # Morphology
        morph_frame = ttk.LabelFrame(scrollable_frame, text="Morphology", padding=10)
        morph_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['morph_size'] = tk.IntVar(value=3)
        self.create_slider(morph_frame, "Structure Size:", 
                          self.parameter_vars['morph_size'], 1, 15, 3)
        
        # Advanced threshold
        adv_thresh_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Threshold", padding=10)
        adv_thresh_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['local_block'] = tk.IntVar(value=35)
        self.create_slider(adv_thresh_frame, "Local Block:", 
                          self.parameter_vars['local_block'], 15, 101, 35)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_multichannel_controls(self, parent):
        """Create multi-channel analysis controls"""
        # Colocalization
        coloc_frame = ttk.LabelFrame(parent, text="Colocalization", padding=15)
        coloc_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['coloc_method'] = tk.StringVar(value='manders')
        methods = [('Manders', 'manders'), ('Pearson', 'pearson'), ('Overlap', 'overlap')]
        
        for text, value in methods:
            ttk.Radiobutton(coloc_frame, text=text,
                           variable=self.parameter_vars['coloc_method'],
                           value=value).pack(anchor='w')
        
        # Channel pairs
        pairs_frame = ttk.LabelFrame(parent, text="Channel Pairs", padding=15)
        pairs_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['pair_ch1_ch2'] = tk.BooleanVar(value=True)
        self.parameter_vars['pair_ch1_ch3'] = tk.BooleanVar(value=True)
        self.parameter_vars['pair_ch2_ch3'] = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(pairs_frame, text="Ch1 ↔ Ch2",
                       variable=self.parameter_vars['pair_ch1_ch2']).pack(anchor='w')
        ttk.Checkbutton(pairs_frame, text="Ch1 ↔ Ch3",
                       variable=self.parameter_vars['pair_ch1_ch3']).pack(anchor='w')
        ttk.Checkbutton(pairs_frame, text="Ch2 ↔ Ch3",
                       variable=self.parameter_vars['pair_ch2_ch3']).pack(anchor='w')
        
        # Distance analysis
        dist_frame = ttk.LabelFrame(parent, text="Distance Analysis", padding=15)
        dist_frame.pack(fill='x', padx=5, pady=5)
        
        self.parameter_vars['distance_threshold'] = tk.DoubleVar(value=10.0)
        self.create_slider(dist_frame, "Max Distance:", 
                          self.parameter_vars['distance_threshold'], 1.0, 50.0, 10.0)
    
    def create_export_controls(self, parent):
        """Create export controls"""
        format_frame = ttk.LabelFrame(parent, text="Export Formats", padding=15)
        format_frame.pack(fill='x', padx=5, pady=5)
        
        self.export_csv = tk.BooleanVar(value=True)
        self.export_json = tk.BooleanVar(value=True)
        self.export_excel = tk.BooleanVar(value=False)
        self.export_images = tk.BooleanVar(value=True)
        self.export_overlays = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(format_frame, text="CSV", variable=self.export_csv).pack(anchor='w')
        ttk.Checkbutton(format_frame, text="JSON", variable=self.export_json).pack(anchor='w')
        ttk.Checkbutton(format_frame, text="Excel", variable=self.export_excel).pack(anchor='w')
        ttk.Checkbutton(format_frame, text="Images", variable=self.export_images).pack(anchor='w')
        ttk.Checkbutton(format_frame, text="Overlays", variable=self.export_overlays).pack(anchor='w')
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(button_frame, text="Export Current Results",
                command=lambda: self.app.export_results('current')).pack(fill='x', pady=2)
        
        # Auto-export
        auto_frame = ttk.Frame(parent)
        auto_frame.pack(fill='x', padx=5, pady=5)
        
        self.auto_export_batch = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_frame, text="Auto-export batch results",
                    variable=self.auto_export_batch).pack(anchor='w')
    
    def create_results_section(self, parent):
        """Create results display section"""
        results_frame = ttk.LabelFrame(parent, text="Results", padding=10)
        results_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Statistics text
        self.stats_text = tk.Text(results_frame, 
                                 height=15, 
                                 width=40,
                                 bg=self.colors['text_bg'],
                                 fg=self.colors['fg'],
                                 font=('Consolas', 9),
                                 wrap='word')
        
        scrollbar = ttk.Scrollbar(results_frame, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initial message
        self.stats_text.insert('1.0', "📊 Results will appear here\n\nLoad images and process to see analysis results.")
    
    def create_progress_section(self, parent):
        """Create progress section for batch processing"""
        self.progress_frame = ttk.LabelFrame(parent, text="Batch Progress", padding=10)
        
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_var)
        progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, 
                                           length=400, 
                                           mode='determinate')
        self.progress_bar.pack(fill='x', pady=(5, 0))
        
    
    def create_slider(self, parent, label, variable, min_val, max_val, default):
        """Helper to create labeled slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        
        ttk.Label(frame, text=label, width=15).pack(side='left')
        
        scale = ttk.Scale(frame, from_=min_val, to=max_val,
                        variable=variable, orient='horizontal')
        scale.pack(side='left', fill='x', expand=True, padx=5)
        
        # Value label
        if isinstance(variable, tk.IntVar):
            value_label = ttk.Label(frame, text=str(int(variable.get())), width=8)
        else:
            value_label = ttk.Label(frame, text=f"{variable.get():.1f}", width=8)
        value_label.pack(side='right')
        
        # Update label when value changes 
        def update_label(*args):
            if isinstance(variable, tk.IntVar):
                value_label.config(text=str(int(variable.get())))
            else:
                value_label.config(text=f"{variable.get():.1f}")
        
        variable.trace('w', update_label)
        
        # Only process on mouse release
        scale.bind('<ButtonRelease-1>', lambda e: self.on_parameter_change())
    
    # Event handlers
    def load_channel_dialog(self, channel):
        """Show dialog to load channel image"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title=f"Select Channel {channel[-1]} Image",
            filetypes=filetypes
        )
        
        if filepath:
            self.app.load_channel(channel, filepath)
    
    def toggle_channel(self, channel):
        """Toggle channel active state"""
        self.app.active_channels[channel] = self.channel_enable_vars[channel].get()
        if self.auto_update_var and self.auto_update_var.get():
            self.app.process_current_image()
        
    def on_mode_change(self):
        """Handle processing mode change"""
        if self.processing_mode.get() == 'batch':
            self.batch_frame.pack(fill='x', pady=(10, 0))
            self.progress_frame.pack(fill='x', pady=(10, 0))
            if hasattr(self, 'process_btn'):
                if hasattr(self.app, 'detected_groups') and self.app.detected_groups:
                    self.process_btn.config(text="🚀 Process Batch", state='normal')
                else:
                    self.process_btn.config(text="🚀 Process Batch", state='disabled')
        else:
            self.batch_frame.pack_forget()
            self.progress_frame.pack_forget()
            if hasattr(self, 'process_btn'):
                active_count = sum(1 for active in self.app.active_channels.values() if active)
                if active_count > 0:
                    if active_count > 1:
                        self.process_btn.config(text="⚡ Process & Analyze", state='normal')
                    else:
                        self.process_btn.config(text="⚡ Process", state='normal')
                else:
                    self.process_btn.config(text="⚡ Process", state='disabled')
    
    def on_parameter_change(self):
        """Handle parameter change with debouncing"""
        if not self.auto_update_var.get():
            return
            
        if not any(self.app.active_channels.values()):
            return
        
        if self.update_timer:
            self.main_window.after_cancel(self.update_timer)
        
        self.update_timer = self.main_window.after(300, self.execute_update)
        
    def execute_update(self):
        """Execute the actual update after debounce delay"""
        self.update_timer = None
        self.app.process_current_image()
    
    def select_batch_folder(self):
        """Select folder for batch processing"""
        folder = filedialog.askdirectory(title="Select Batch Processing Folder")
        if folder:
            self.batch_status.set(f"Folder: {os.path.basename(folder)}")
            self.app.select_batch_folder(folder)  # Pass the folder to the app
        else:
            self.batch_status.set("No folder selected")
    
    # Update methods
    def update_channel_status(self, channel, filepath):
        """Update channel status display"""
        filename = os.path.basename(filepath)
        self.channel_status_vars[channel].set(filename)
        self.channel_enable_vars[channel].set(True)
    
    def update_pattern_display(self, pattern):
        """Update pattern detection display"""
        if pattern:
            pattern_text = f"{pattern['prefix']}[{'/'.join(pattern['channel_ids'].values())}]{pattern['suffix']}"
            self.pattern_var.set(pattern_text)
    
    def update_preview_display(self, app):
        """Update the preview display - Fixed version to prevent histogram disappearing"""
        view = self.preview_var.get()
        
        print(f"Updating preview display to: {view}")
        
        for i, ax_row in enumerate(self.axes):
            for j, ax in enumerate(ax_row):
                # Store the current figure and position info before clearing
                fig = ax.figure
                pos = ax.get_position()                
                ax.clear()
                ax.set_facecolor(self.colors['bg'])
                ax.set_position(pos)
        
        if view == 'composite':
            self.display_composite(app)
        else:
            self.display_single_channel(app, view)
        
        try:
            self.fig.tight_layout(pad=2.0)
        except:
            pass
        
        try:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            self.canvas.draw()
        except Exception as e:
            print(f"Canvas draw error (non-critical): {e}")
            # Fallback
            try:
                self.canvas.draw_idle()
            except:
                pass
        
        print(f"Preview display updated to {view}")

    def _style_axis(self, ax):
        """Helper method to consistently style axes"""
        ax.tick_params(colors=self.colors['fg'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.colors['panel'])
            spine.set_linewidth(0.5)
        ax.grid(False)
        ax.axis('on')

    def display_single_channel(self, app, channel):
        """Display single channel preview with reliable histogram - FIXED VERSION"""
        if channel not in app.channels or app.channels[channel] is None:
            for ax_row in self.axes:
                for ax in ax_row:
                    ax.clear()
                    ax.axis('off')
            self.axes[0, 1].text(0.5, 0.5, f"No {channel} loaded",
                                ha='center', va='center',
                                transform=self.axes[0, 1].transAxes,
                                color=self.colors['fg'])
            return
        
        image_data = app.channels[channel].copy().flatten()
        
        # Original image
        ax = self.axes[0, 0]
        ax.clear()
        ax.imshow(app.channels[channel], cmap='gray')
        ax.set_title('Original', color=self.colors['fg'])
        self._style_axis(ax)
        
        # Processed image 
        ax = self.axes[0, 1] 
        ax.clear()
        if channel in app.processed_channels and app.processed_channels[channel] is not None:
            ax.imshow(app.processed_channels[channel], cmap='gray')
            ax.set_title('Processed', color=self.colors['fg'])
            self._style_axis(ax)
        else:
            ax.axis('off')
        
        # Binary mask
        ax = self.axes[0, 2]
        ax.clear()
        if channel in app.binary_masks and app.binary_masks[channel] is not None:
            ax.imshow(app.binary_masks[channel], cmap='gray')
            ax.set_title('Binary', color=self.colors['fg'])
            self._style_axis(ax)
        else:
            ax.axis('off')
                
        # Labeled objects
        ax = self.axes[1, 0]
        ax.clear()
        if app.labeled_objects[channel] is not None:
            ax.imshow(app.labeled_objects[channel], cmap='tab20')
            n_objects = np.max(app.labeled_objects[channel])
            ax.set_title(f'Objects: {n_objects}', color=self.colors['fg'])
            self._style_axis(ax)
        else:
            ax.axis('off')
        
        ax_hist = self.axes[1, 1]
        
        ax_hist.clear()
        ax_hist.cla()
        
        try:
            # Validate data before creating histogram
            if len(image_data) == 0 or np.all(np.isnan(image_data)):
                raise ValueError("No valid image data for histogram")
            
            # Remove any NaN or infinite values
            valid_data = image_data[np.isfinite(image_data)]
            
            if len(valid_data) == 0:
                raise ValueError("No finite values in image data")
            
            # Calculate appropriate number of bins
            data_range = np.max(valid_data) - np.min(valid_data)
            if data_range == 0:
                n_bins = 1
            else:
                n_bins = min(50, max(10, int(len(np.unique(valid_data)))))
            
            # Create histogram with error handling
            counts, bins, patches = ax_hist.hist(
                valid_data,
                bins=n_bins,
                alpha=0.8,
                color=self.colors[f'{channel}_color'],
                edgecolor='black',  # Add edge for visibility
                linewidth=0.5,
                density=False,  # Use counts instead of density
                histtype='bar'
            )
            
            # Verify the histogram was created successfully
            if len(patches) == 0:
                raise ValueError("Histogram creation failed - no patches created")
            
            # Force visibility of all patches
            for patch in patches:
                patch.set_visible(True)
                patch.set_alpha(0.8)
                patch.set_facecolor(self.colors[f'{channel}_color'])
            
            # Set up axis properties
            ax_hist.set_facecolor(self.colors['bg'])
            ax_hist.set_title('Histogram', color=self.colors['fg'], fontsize=10)
            ax_hist.set_xlabel('Intensity', color=self.colors['fg'], fontsize=9)
            ax_hist.set_ylabel('Count', color=self.colors['fg'], fontsize=9)
            
            # Apply consistent styling
            self._style_axis(ax_hist)
            
            # Set explicit axis limits to ensure visibility
            x_margin = data_range * 0.02 if data_range > 0 else 0.01
            ax_hist.set_xlim([np.min(valid_data) - x_margin, np.max(valid_data) + x_margin])
            
            y_max = np.max(counts)
            ax_hist.set_ylim([0, y_max * 1.1 if y_max > 0 else 1])
            
            # Force the axis to recalculate its layout
            ax_hist.relim()
            ax_hist.autoscale_view(tight=False)
            
            print(f"Histogram successfully created for {channel}: {len(counts)} bins, max count: {np.max(counts)}")
            
        except Exception as e:
            print(f"Histogram creation error: {e}")
            # Fallback - create simple text message
            ax_hist.clear()
            ax_hist.text(0.5, 0.5, f"Histogram\n{channel}\nData range: {np.min(image_data):.3f} - {np.max(image_data):.3f}",
                        ha='center', va='center',
                        transform=ax_hist.transAxes,
                        color=self.colors['fg'],
                        fontsize=9)
            ax_hist.set_facecolor(self.colors['bg'])
            self._style_axis(ax_hist)
        
        # Clear unused subplot
        ax = self.axes[1, 2]
        ax.clear()
        ax.axis('off')

    def display_composite(self, app):
        """Display multi-channel composite and colocalization view"""
        active = [ch for ch in ['ch1', 'ch2', 'ch3'] 
                if app.active_channels[ch] and app.channels[ch] is not None]
        
        print(f"Display composite called with active channels: {active}")
        
        if not active:
            self.axes[0, 1].text(0.5, 0.5, "No active channels",
                                ha='center', va='center',
                                transform=self.axes[0, 1].transAxes,
                                color=self.colors['fg'])
            return
        
        # Check if we have multichannel results AND user has selected channel pairs
        params = self.get_multichannel_parameters()
        has_selected_pairs = (
            (params['pairs']['ch1_ch2'] and 'ch1' in active and 'ch2' in active) or
            (params['pairs']['ch1_ch3'] and 'ch1' in active and 'ch3' in active) or
            (params['pairs']['ch2_ch3'] and 'ch2' in active and 'ch3' in active)
        )
        
        print(f"Has multichannel results: {hasattr(app, 'multichannel_results') and bool(app.multichannel_results)}")
        print(f"Has selected pairs: {has_selected_pairs}")
        
        # Show colocalization analysis if we have results and user selected pairs
        if (hasattr(app, 'multichannel_results') and app.multichannel_results and 
            len(active) >= 2 and has_selected_pairs):
            print("Displaying colocalization analysis")
            self.display_colocalization_analysis(app, active)
        else:
            print("Displaying standard composite")
            self.display_standard_composite(app, active)
        
    def display_standard_composite(self, app, active):
        """Display standard RGB composite view"""
        # Build RGB composite
        shape = app.channels[active[0]].shape
        composite = np.zeros((*shape, 3))
        
        # Map channels to RGB
        color_map = {'ch1': 0, 'ch2': 1, 'ch3': 2}
        for ch in active:
            if ch in color_map:
                channel_data = app.channels[ch]
                if channel_data.max() > 1.0:
                    channel_data = channel_data / channel_data.max()
                composite[:, :, color_map[ch]] = channel_data
        
        # Display composite
        self.axes[0, 1].imshow(composite)
        self.axes[0, 1].set_title('RGB Composite', color=self.colors['fg'])
        self.axes[0, 1].axis('on')
        
        # Show individual channels with proper colormaps
        colormaps = {'ch1': 'Reds', 'ch2': 'Greens', 'ch3': 'Blues'}
        for i, ch in enumerate(active[:3]):
            if i < 3:
                ax = self.axes[1, i]
                ax.clear()
                ax.imshow(app.channels[ch], cmap=colormaps.get(ch, 'gray'))
                ax.set_title(self.channel_names[ch], color=self.colors['fg'])
                ax.axis('on')

    def apply_channel_color_to_image(self, image, channel):
        """Apply channel-specific color tinting to grayscale image"""
        # Convert grayscale to RGB
        rgb_image = np.stack([image, image, image], axis=2)
        
        if channel == 'ch1':  # Red channel
            rgb_image[:, :, [1, 2]] *= 0.3  # Reduce green and blue
        elif channel == 'ch2':  # Green channel  
            rgb_image[:, :, [0, 2]] *= 0.3  # Reduce red and blue
        elif channel == 'ch3':  # Blue channel
            rgb_image[:, :, [0, 1]] *= 0.3  # Reduce red and green
        
        return rgb_image

    def display_single_pair_statistics(self, app, pair):
        """Display statistics for a single channel pair in remaining space"""
        ch1, ch2 = pair
        pair_key = f"{ch1}_{ch2}"
        
        # Find an available subplot - prefer [0,2] if not used
        if len([ch for ch in ['ch1', 'ch2', 'ch3'] if app.active_channels[ch]]) <= 2:
            ax_stats = self.axes[0, 2]
            ax_stats.clear()
            ax_stats.axis('off')
            
            if pair_key in app.multichannel_results:
                coloc_data = app.multichannel_results[pair_key]['colocalization']
                
                stats_text = f"Colocalization Analysis\n"
                stats_text += f"{self.channel_names[ch1]} + {self.channel_names[ch2]}\n\n"
                stats_text += f"Overlap: {coloc_data.get('overlap_percentage', 0):.1f}%\n"
                stats_text += f"Manders M1: {coloc_data.get('manders_m1', 0):.3f}\n"
                stats_text += f"Manders M2: {coloc_data.get('manders_m2', 0):.3f}\n"
                stats_text += f"Pearson r: {coloc_data.get('pearson_correlation', 0):.3f}\n\n"
                stats_text += "Color Legend:\n"
                stats_text += "RED = Colocalized\n"
                stats_text += "CYAN = Ch1 only\n" 
                stats_text += "BLUE = Ch2 only"
                
                ax_stats.text(0.05, 0.95, stats_text,
                            transform=ax_stats.transAxes,
                            fontsize=8,
                            color=self.colors['fg'],
                            verticalalignment='top',
                            fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                    facecolor=self.colors['panel'], 
                                    alpha=0.8))

    def display_multiple_pair_statistics(self, app, pairs):
        """Display statistics for multiple channel pairs"""
        stats_text = "Multi-Channel Analysis\n\n"
        
        for ch1, ch2 in pairs:
            pair_key = f"{ch1}_{ch2}"
            if pair_key in app.multichannel_results:
                coloc_data = app.multichannel_results[pair_key]['colocalization']
                stats_text += f"{ch1}+{ch2}: "
                stats_text += f"{coloc_data.get('overlap_percentage', 0):.1f}% overlap, "
                stats_text += f"M1={coloc_data.get('manders_m1', 0):.2f}, "
                stats_text += f"M2={coloc_data.get('manders_m2', 0):.2f}\n"
        
        # Display in bottom-right if available
        if hasattr(self.axes[1], '__len__') and len(self.axes[1]) > 2:
            ax_stats = self.axes[1, 2]
            ax_stats.clear() 
            ax_stats.axis('off')
            ax_stats.text(0.05, 0.95, stats_text,
                        transform=ax_stats.transAxes,
                        fontsize=8,
                        color=self.colors['fg'],
                        verticalalignment='top',
                        fontfamily='monospace')
        
    def display_colocalization_analysis(self, app, active):
        """Display colocalization overlay analysis with user-selected channel pairs"""
        
        # Get user-selected channel pairs from Multi-Channel settings
        params = self.get_multichannel_parameters()
        selected_pairs = []
        
        if params['pairs']['ch1_ch2'] and 'ch1' in active and 'ch2' in active:
            selected_pairs.append(('ch1', 'ch2'))
        if params['pairs']['ch1_ch3'] and 'ch1' in active and 'ch3' in active:
            selected_pairs.append(('ch1', 'ch3'))
        if params['pairs']['ch2_ch3'] and 'ch2' in active and 'ch3' in active:
            selected_pairs.append(('ch2', 'ch3'))
        
        if not selected_pairs or not hasattr(app, 'multichannel_results') or not app.multichannel_results:
            self.display_standard_composite(app, active)
            return
        
        # Display the first selected pair as main colocalization overlay
        primary_pair = selected_pairs[0]
        ch1, ch2 = primary_pair
        
        # Create colocalization overlay for the primary pair
        coloc_overlay = self.create_colocalization_overlay_for_display(
            app.processed_channels.get(ch1),
            app.processed_channels.get(ch2),
            app.binary_masks.get(ch1),
            app.binary_masks.get(ch2),
            ch1, ch2
        )
        
        # Main colocalization display
        ax_main = self.axes[0, 1]
        ax_main.clear()
        if coloc_overlay is not None:
            ax_main.imshow(coloc_overlay)
            ax_main.set_title(f'Colocalization: {self.channel_names[ch1]} + {self.channel_names[ch2]}', 
                            color=self.colors['fg'], fontsize=10)
            ax_main.axis('on')
        
        display_positions = [(0, 0), (0, 2)]  # Skip middle slot used for colocalization
        
        for i, ch in enumerate(active[:2]):  # Show up to 2 individual channels in top row
            if i < len(display_positions):
                pos = display_positions[i]
                ax = self.axes[pos[0], pos[1]]
                ax.clear()
                
                # Show original image with channel color overlay
                if app.channels.get(ch) is not None:
                    # Create colored version of the original image
                    colored_image = self.apply_channel_color_to_image(app.channels[ch], ch)
                    ax.imshow(colored_image)
                    ax.set_title(f'{self.channel_names[ch]}', color=self.colors['fg'], fontsize=10)
                    ax.axis('on')
        
        # Bottom row: Show all active channels with their detected objects
        for i, ch in enumerate(active[:3]):
            if i < 3:
                ax = self.axes[1, i]
                ax.clear()
                
                if app.binary_masks.get(ch) is not None and app.channels.get(ch) is not None:
                    # Create overlay of detected objects on original image
                    overlay = self.create_single_channel_overlay_for_display(
                        app.channels[ch],
                        app.binary_masks[ch],
                        ch
                    )
                    ax.imshow(overlay)
                    
                    # Add object count to title
                    n_objects = 0
                    if app.labeled_objects.get(ch) is not None:
                        n_objects = np.max(app.labeled_objects[ch])
                    
                    ax.set_title(f'{self.channel_names[ch]}: {n_objects} objects', 
                            color=self.colors['fg'], fontsize=9)
                else:
                    # Just show the original channel
                    if app.channels.get(ch) is not None:
                        ax.imshow(app.channels[ch], cmap='gray')
                        ax.set_title(self.channel_names[ch], color=self.colors['fg'], fontsize=9)
                ax.axis('on')
        
        if len(selected_pairs) > 1:
            self.display_multiple_pair_statistics(app, selected_pairs)
        else:
            self.display_single_pair_statistics(app, primary_pair)

    def create_colocalization_overlay_for_display(self, img1, img2, mask1, mask2, ch1, ch2):
        """Create colocalization overlay for GUI display"""
        if img1 is None or mask1 is None or mask2 is None:
            return None
        
        # Ensure compatible shapes
        if img1.shape != img2.shape or mask1.shape != mask2.shape:
            min_h = min(img1.shape[0], img2.shape[0], mask1.shape[0], mask2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1], mask1.shape[1], mask2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w] 
            mask1 = mask1[:min_h, :min_w]
            mask2 = mask2[:min_h, :min_w]
        
        overlay = np.stack([img1, img1, img1], axis=2)  # RGB from grayscale
        
        # Channel 1 objects in cyan (exclusive regions)
        ch1_only = mask1 & ~mask2
        overlay[ch1_only] = [0, 1, 1]  # Cyan
        
        # Overlapping regions in bright red
        overlap = mask1 & mask2
        overlay[overlap] = [1, 0, 0]  # Red
        
        # Channel 2 only regions in blue
        ch2_only = mask2 & ~mask1
        overlay[ch2_only] = [0, 0.4, 1]  # Light blue
        
        return overlay

    def create_single_channel_overlay_for_display(self, image, mask, channel):
        """Create single channel overlay for display"""
        if image is None or mask is None:
            return image
        
        # Create RGB overlay
        overlay = np.stack([image, image, image], axis=2)
        
        # Apply channel color to mask regions
        channel_colors = {
            'ch1': [1, 0.3, 0.3],    # Red tint
            'ch2': [0.3, 1, 0.3],    # Green tint  
            'ch3': [0.3, 0.3, 1]     # Blue tint
        }
        
        color = channel_colors.get(channel, [1, 1, 1])
        overlay[mask] = color
        
        return overlay
    def update_statistics(self, quant_results, multi_results):
        """Update statistics display"""
        self.stats_text.delete('1.0', tk.END)
        
        text = "📊 ANALYSIS RESULTS\n" + "="*40 + "\n\n"
        
        # Single channel results
        if quant_results:
            for channel, results in quant_results.items():
                if results:
                    text += f"🔍 {self.channel_names[channel]}:\n"
                    text += f"  Objects: {results['total_objects']}\n"
                    text += f"  Coverage: {results['coverage_percent']:.1f}%\n"
                    text += f"  Mean Area: {results['mean_object_area']:.1f} px\n"
                    text += f"  Total Signal: {results['total_fluorescence']:.1f}\n\n"
        
        # Multi-channel results
        if multi_results:
            text += "🔗 MULTI-CHANNEL ANALYSIS\n" + "="*40 + "\n\n"
            
            for pair_key, results in multi_results.items():
                ch1, ch2 = results['channels']
                coloc = results['colocalization']
                
                text += f"{self.channel_names[ch1]} ↔ {self.channel_names[ch2]}:\n"
                text += f"  Overlap: {coloc['overlap_percentage']:.1f}%\n"
                text += f"  Manders M1: {coloc['manders_m1']:.3f}\n"
                text += f"  Manders M2: {coloc['manders_m2']:.3f}\n"
                text += f"  Pearson r: {coloc['pearson_correlation']:.3f}\n\n"
        
        self.stats_text.insert('1.0', text)
    
    def update_batch_progress(self, current, total, message):
        """Update batch processing progress"""
        if total > 0:
            progress = (current / total) * 100
            self.progress_bar['value'] = progress
            self.progress_var.set(f"{message} ({current}/{total})")
    
    def enable_processing_buttons(self):
        """Enable processing buttons when channels are loaded"""
        if self.processing_mode.get() == 'batch':
            # For batch mode, check if groups are detected
            if hasattr(self.app, 'detected_groups') and self.app.detected_groups:
                self.process_btn.config(state='normal', text="🚀 Process Batch")
            else:
                self.process_btn.config(state='disabled', text="🚀 Process Batch")
        else:
            # For single image mode
            self.process_btn.config(state='normal')
            active_count = sum(1 for active in self.app.active_channels.values() if active)
            if active_count > 1:
                self.process_btn.config(text="⚡ Process & Analyze")
            else:
                self.process_btn.config(text="⚡ Process")
    
    def enable_batch_folder_selection(self):
        """Enable batch folder selection when pattern learned"""
        self.batch_button.config(state='normal')
    
    def enable_batch_processing(self, n_files):
        """Enable batch processing when groups are detected"""
        self.batch_status.set(f"Ready: {n_files} groups found")
        if self.processing_mode.get() == 'batch':
            self.process_btn.config(state='normal', text="🚀 Process Batch")
    
    def batch_processing_complete(self, n_processed):
        """Update UI when batch processing completes"""
        self.progress_var.set(f"Complete! Processed {n_processed} groups")
        self.show_info(f"Batch processing complete!\n{n_processed} image groups processed.")
    
    def reset_display(self):
        """Reset all displays"""
        for ch in ['ch1', 'ch2', 'ch3']:
            self.channel_status_vars[ch].set("No file loaded")
            self.channel_enable_vars[ch].set(False)
        
        self.pattern_var.set("Load 2+ channels to learn pattern")
        self.batch_status.set("")
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', "📊 Results will appear here")
        
        self.update_preview_display(self.app)
    
    # Parameter getters
    def get_processing_parameters(self):
        """Get current processing parameters"""
        return {
            'preprocessing': {
                'gaussian': self.parameter_vars.get('gaussian', tk.BooleanVar()).get(),
                'gaussian_sigma': self.parameter_vars.get('gaussian_sigma', tk.DoubleVar(value=1.0)).get(),
                'median': self.parameter_vars.get('median', tk.BooleanVar()).get(),
                'median_size': self.parameter_vars.get('median_size', tk.IntVar(value=3)).get(),
                'clahe': self.parameter_vars.get('clahe', tk.BooleanVar()).get(),
                'clahe_clip': self.parameter_vars.get('clahe_clip', tk.DoubleVar(value=2.0)).get(),
                'gamma': self.parameter_vars.get('gamma', tk.DoubleVar(value=1.0)).get(),
                'contrast': self.parameter_vars.get('contrast', tk.DoubleVar(value=1.0)).get(),
                'brightness': self.parameter_vars.get('brightness', tk.IntVar(value=0)).get(),
                'morphology': self.parameter_vars.get('morphology', tk.BooleanVar()).get(),
                'morph_size': self.parameter_vars.get('morph_size', tk.IntVar(value=3)).get()
            },
            'thresholding': {
                'method': self.threshold_method.get(),
                'manual_threshold': self.manual_thresh_var.get(),
                'local_block_size': self.parameter_vars.get('local_block', tk.IntVar(value=35)).get(),
                'min_size': self.parameter_vars.get('min_size', tk.IntVar(value=50)).get(),
                'max_size': self.parameter_vars.get('max_size', tk.IntVar(value=10000)).get()
            }
        }
    
    def get_analysis_mode(self):
        """Get current analysis mode"""
        return self.mode_var.get()
    
    def get_multichannel_parameters(self):
        """Get multi-channel analysis parameters"""
        return {
            'method': self.parameter_vars.get('coloc_method', tk.StringVar(value='manders')).get(),
            'distance_threshold': self.parameter_vars.get('distance_threshold', tk.DoubleVar(value=10.0)).get(),
            'pairs': {
                'ch1_ch2': self.parameter_vars.get('pair_ch1_ch2', tk.BooleanVar(value=True)).get(),
                'ch1_ch3': self.parameter_vars.get('pair_ch1_ch3', tk.BooleanVar(value=True)).get(),
                'ch2_ch3': self.parameter_vars.get('pair_ch2_ch3', tk.BooleanVar(value=True)).get()
            }
        }
    
    def get_export_options(self):
        """Get export options"""
        return {
            'export_csv': self.export_csv.get(),
            'export_json': self.export_json.get(),
            'export_excel': self.export_excel.get(),
            'export_images': self.export_images.get(),
            'export_overlays': self.export_overlays.get()
        }
    
    def get_save_filepath(self):
        """Get save file path from user"""
        return filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
    
    # Dialog methods
    def show_info(self, message):
        """Show info dialog"""
        messagebox.showinfo("Information", message)
    
    def show_warning(self, message):
        """Show warning dialog"""
        messagebox.showwarning("Warning", message)
    
    def show_error(self, message):
        """Show error dialog"""
        messagebox.showerror("Error", message)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """FluoroQuant v2.0
        
Multi-Channel Fluorescence Analysis Platform

Features:
• Multi-channel image processing
• Advanced thresholding algorithms
• Colocalization analysis
• Batch processing with pattern detection
• Comprehensive export options
"""
        
        messagebox.showinfo("About FluoroQuant", about_text)
    
    def show_guide(self):
        """Show user guide"""
        guide_text = """Quick Start Guide:

1. Load Images:
   - Click "Load Image" for each channel
   - Supports TIFF, PNG, JPG formats

2. Processing:
   - Adjust parameters in Quick/Advanced tabs
   - Click "Process" or enable Auto-update

3. Analysis:
   - View results in the statistics panel
   - Switch between channel views

4. Batch Processing:
   - Load 2+ channels to learn naming pattern
   - Select folder with matching files
   - Click "Start Batch"

5. Export:
   - Choose formats in Export tab
   - Click "Export Current" or "Export Batch"

Tips:
• Use keyboard shortcuts for efficiency
• Save parameter presets for consistency
• Check Multi-Channel tab for colocalization"""
        
        messagebox.showinfo("User Guide", guide_text)
    
    def show_batch_preview(self, detected_groups):
        """Show preview of detected file groups"""
        # Create preview window
        preview_window = tk.Toplevel(self.main_window)
        preview_window.title("Batch Files Preview")
        preview_window.geometry("800x600")
        preview_window.configure(bg=self.colors['bg'])
        
        # Apply dark theme to toplevel
        preview_window.tk_setPalette(background=self.colors['bg'],
                                    foreground=self.colors['fg'])
        
        # Main frame
        main_frame = ttk.Frame(preview_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Detected File Groups",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Create treeview
        columns = ('Group', 'Channel 1', 'Channel 2', 'Channel 3')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        tree.column('Group', width=150)
        tree.column('Channel 1', width=200)
        tree.column('Channel 2', width=200)
        tree.column('Channel 3', width=200)
        
        for col in columns:
            tree.heading(col, text=col)
        
        # Style treeview for dark theme
        style = ttk.Style()
        style.configure("Treeview",
                       background=self.colors['panel'],
                       foreground=self.colors['fg'],
                       fieldbackground=self.colors['panel'])
        style.configure("Treeview.Heading",
                       background=self.colors['button_bg'],
                       foreground=self.colors['fg'])
        
        # Populate tree
        for i, (group_name, channels) in enumerate(detected_groups.items()):
            # Handle None values properly
            ch1 = os.path.basename(channels['ch1']) if channels.get('ch1') else '—'
            ch2 = os.path.basename(channels['ch2']) if channels.get('ch2') else '—'
            ch3 = os.path.basename(channels['ch3']) if channels.get('ch3') else '—'
            
            tree.insert('', 'end', values=(group_name, ch1, ch2, ch3))
        
        tree.pack(side='left', fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Summary
        summary_label = ttk.Label(main_frame,
                                 text=f"Total: {len(detected_groups)} image groups",
                                 font=('Arial', 11, 'bold'))
        summary_label.pack(pady=(10, 0))
        
        # Close button
        close_btn = ttk.Button(main_frame, text="Close", 
                              command=preview_window.destroy)
        close_btn.pack(pady=(10, 0))