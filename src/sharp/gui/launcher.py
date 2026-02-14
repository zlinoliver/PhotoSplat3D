#!/usr/bin/env python3
"""Standalone launcher for SHARP GUI app.

This is used as the entry point when building standalone applications.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Launch the SHARP GUI application."""
    try:
        from sharp.gui.main_window import SharpGUI

        app = SharpGUI()
        app.mainloop()
    except Exception as e:
        logging.error(f"Failed to start SHARP GUI: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    try:
        import multiprocessing as mp

        mp.freeze_support()
    except Exception:
        pass
    main()
