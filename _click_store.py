"""Child-process helper: captures mouse clicks and writes them to a file.

The parent polls the file's mtime each loop; when it changes, it reads
the latest click coordinates. No queues, no Manager, no IPC complexity.

Imported by the child QtProcess, not used directly by the parent.
"""
_click_file = None
_vb = None


def setup(win, click_file):
    """Register a click handler on the plot window.

    Args:
        win:        PlotWidget (real object, lives in this process)
        click_file: Absolute path string to write click coords to
    """
    global _click_file, _vb
    _click_file = click_file
    _vb = win.getViewBox()
    win.scene().sigMouseClicked.connect(_on_click)


def _on_click(event):
    if _click_file is None or _vb is None:
        return
    pos = _vb.mapSceneToView(event.scenePos())
    try:
        with open(_click_file, 'w') as f:
            f.write(f'{float(pos.x())},{float(pos.y())}\n')
    except Exception:
        pass
