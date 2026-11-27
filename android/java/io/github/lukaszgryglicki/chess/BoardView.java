package io.github.lukaszgryglicki.chess;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;

import java.io.File;

/** The board: draws what the game reports and passes the finger on to it. */
final class BoardView extends View {
    private static final String PIECES = "PNBRQKpnbrqk";
    private static final String[] FILES = {"wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"};
    private static final int LIGHT = 0xFFF0D9B5, DARK = 0xFFB58863;
    private static final int LAST = 0x80F6E05A, SELECTED = 0xA05FA83C, TARGET = 0x7050A0E0;
    private static final int PANEL = 0xFFFFFAF0, LINE = 0xFF202020;

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Bitmap[] sprites = new Bitmap[12];
    private final Rect dst = new Rect();
    private final int touchSlop;

    private Engine engine;
    private Runnable wake;        // what a touch does while there is no game process
    private File pictures;        // where the game draws the piece pictures, a directory per size
    private int unit = 64;        // the game's units to a square
    private int cell, ox, oy;     // a square in pixels and the board's corner
    private int asked;            // the square size the pictures were asked in
    private String board;
    private int sel = -1, from = -1, to = -1, promo = -1, held = -1;
    private boolean tracking, moved;
    private float downX, downY, fx, fy;

    BoardView(Context c) {
        super(c);
        touchSlop = ViewConfiguration.get(c).getScaledTouchSlop();
        paint.setTypeface(Typeface.DEFAULT_BOLD);
    }

    void setWake(Runnable r) {
        wake = r;
    }

    void setPictures(File dir) {
        pictures = dir;
    }

    /** A game has started; unit is how many of its units make a square. */
    void attach(Engine e, int unit) {
        engine = e;
        this.unit = unit;
        for (Bitmap b : sprites) {
            if (b == null) {
                asked = 0; // the game before this one never sent them all
            }
        }
        askSprites();
    }

    /** The game is gone; clear also takes its last picture off the board. */
    void detach(boolean clear) {
        engine = null;
        tracking = false;
        held = -1;
        if (clear) {
            board = null;
            sel = from = to = promo = -1;
        }
        invalidate();
    }

    /**
     * The piece pictures in the size of the screen's squares: those an earlier
     * game drew are still in their directory, else this game is asked for them.
     */
    private void askSprites() {
        if (cell <= 0 || cell == asked || pictures == null) {
            return;
        }
        File dir = new File(pictures, "pieces-" + cell);
        if (load(dir)) {
            asked = cell;
        } else if (engine != null) {
            asked = cell;
            engine.send("size " + cell + " " + dir.getPath());
        }
    }

    /** The game has drawn the pictures, n pixels wide, into dir. */
    void sprites(int n, String dir) {
        if (n == cell) {
            load(new File(dir));
        }
    }

    private boolean load(File dir) {
        Bitmap[] got = new Bitmap[FILES.length];
        for (int i = 0; i < got.length; i++) {
            got[i] = BitmapFactory.decodeFile(new File(dir, FILES[i] + ".png").getPath());
            if (got[i] == null || got[i].getWidth() != cell) {
                return false;
            }
        }
        System.arraycopy(got, 0, sprites, 0, got.length);
        invalidate();
        return true;
    }

    void state(String board, int sel, int from, int to, int promo, int held) {
        this.board = board;
        this.sel = sel;
        this.from = from;
        this.to = to;
        this.promo = promo;
        this.held = held;
        invalidate();
    }

    @Override
    protected void onMeasure(int wSpec, int hSpec) {
        int w = MeasureSpec.getSize(wSpec), h = MeasureSpec.getSize(hSpec);
        int s = Math.min(w, h);
        if (MeasureSpec.getMode(wSpec) == MeasureSpec.UNSPECIFIED) {
            s = h;
        } else if (MeasureSpec.getMode(hSpec) == MeasureSpec.UNSPECIFIED) {
            s = w;
        }
        setMeasuredDimension(s, s);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldW, int oldH) {
        cell = Math.min(w, h) / 8;
        ox = (w - 8 * cell) / 2;
        oy = (h - 8 * cell) / 2;
        askSprites();
    }

    private int squareX(int sq) {
        return ox + (sq & 7) * cell;
    }

    private int squareY(int sq) {
        return oy + (7 - (sq >> 3)) * cell;
    }

    private int squareAt(float x, float y) {
        int f = (int) Math.floor((x - ox) / cell), r = 7 - (int) Math.floor((y - oy) / cell);
        return f < 0 || f > 7 || r < 0 || r > 7 ? -1 : r * 8 + f;
    }

    private void tint(Canvas c, int sq, int color) {
        if (sq >= 0 && sq < 64) {
            paint.setColor(color);
            c.drawRect(squareX(sq), squareY(sq), squareX(sq) + cell, squareY(sq) + cell, paint);
        }
    }

    private void piece(Canvas c, char p, int x, int y, int size) {
        int i = PIECES.indexOf(p);
        if (i >= 0 && sprites[i] != null) {
            dst.set(x, y, x + size, y + size);
            c.drawBitmap(sprites[i], null, dst, paint);
        }
    }

    @Override
    protected void onDraw(Canvas c) {
        if (cell <= 0) {
            return;
        }
        paint.setStyle(Paint.Style.FILL);
        for (int sq = 0; sq < 64; sq++) {
            tint(c, sq, (((sq & 7) + (sq >> 3)) & 1) == 0 ? DARK : LIGHT);
        }
        tint(c, from, LAST);
        tint(c, to, LAST);
        tint(c, sel, SELECTED);
        if (held >= 0 && tracking && squareAt(fx, fy) != held) {
            tint(c, squareAt(fx, fy), TARGET);
        }
        paint.setTextSize(cell * 0.17f);
        float pad = cell * 0.04f;
        for (int i = 0; i < 8; i++) { // files along the bottom, ranks down the left
            paint.setColor((i & 1) == 0 ? LIGHT : DARK);
            paint.setTextAlign(Paint.Align.RIGHT);
            c.drawText(String.valueOf((char) ('a' + i)), squareX(i) + cell - pad, squareY(i) + cell - pad, paint);
            paint.setColor((i & 1) == 0 ? LIGHT : DARK);
            paint.setTextAlign(Paint.Align.LEFT);
            c.drawText(String.valueOf((char) ('1' + i)), squareX(i * 8) + pad, squareY(i * 8) + pad - paint.ascent(), paint);
        }
        if (board == null) {
            return;
        }
        paint.setColor(0xFFFFFFFF);
        for (int sq = 0; sq < 64; sq++) {
            if (sq != held) {
                piece(c, board.charAt(sq), squareX(sq), squareY(sq), cell);
            }
        }
        if (promo >= 0) { // the chooser lies where the game expects the answer
            int x0 = ox + 2 * cell, y0 = oy + 3 * cell + cell / 2, m = cell / 8;
            paint.setColor(PANEL);
            c.drawRect(x0 - m, y0 - m, x0 + 4 * cell + m, y0 + cell + m, paint);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(Math.max(2, cell / 32f));
            paint.setColor(LINE);
            c.drawRect(x0 - m, y0 - m, x0 + 4 * cell + m, y0 + cell + m, paint);
            paint.setStyle(Paint.Style.FILL);
            for (int i = 0; i < 4; i++) {
                piece(c, "QRBN".charAt(i), x0 + i * cell, y0, cell);
            }
        }
        if (held >= 0) { // the piece in hand: larger, and above the finger that would hide it
            int size = cell * 3 / 2;
            piece(c, board.charAt(held), Math.round(fx) - size / 2, Math.round(fy) - size / 2 - cell * 3 / 4, size);
        }
    }

    private int units(float pixels) {
        return (int) Math.floor(pixels * unit / cell);
    }

    private String at(float x, float y) {
        return " " + units(x - ox) + " " + units(y - oy);
    }

    @Override
    public boolean onTouchEvent(MotionEvent ev) {
        if (engine == null || cell <= 0) {
            if (ev.getActionMasked() == MotionEvent.ACTION_DOWN && wake != null) {
                wake.run();
            }
            return false;
        }
        switch (ev.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                downX = fx = ev.getX();
                downY = fy = ev.getY();
                tracking = true;
                moved = false;
                getParent().requestDisallowInterceptTouchEvent(true);
                engine.send("d" + at(downX, downY));
                break;
            case MotionEvent.ACTION_MOVE:
                if (tracking) {
                    fx = ev.getX();
                    fy = ev.getY();
                    // a finger never rests quite still: only a real move lifts the piece
                    if (!moved && Math.hypot(fx - downX, fy - downY) > Math.max(touchSlop, cell / 5f)) {
                        moved = true;
                        engine.send("m" + at(fx, fy));
                    }
                    if (moved) {
                        invalidate();
                    }
                }
                break;
            case MotionEvent.ACTION_UP:
                if (tracking) {
                    tracking = false;
                    fx = ev.getX();
                    fy = ev.getY();
                    engine.send("u" + (moved ? at(fx, fy) : at(downX, downY)));
                    invalidate();
                }
                break;
            case MotionEvent.ACTION_POINTER_DOWN: // a second finger: whose move would it be?
            case MotionEvent.ACTION_CANCEL:
                if (tracking) {
                    tracking = false;
                    engine.send("cancel");
                    invalidate();
                }
                break;
            default:
                break;
        }
        return true;
    }
}
