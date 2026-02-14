"""CLI command to launch GUI.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click


@click.command()
def gui_cli():
    """Launch the SHARP GUI application."""
    from sharp.gui.main_window import SharpGUI

    app = SharpGUI()
    app.mainloop()
