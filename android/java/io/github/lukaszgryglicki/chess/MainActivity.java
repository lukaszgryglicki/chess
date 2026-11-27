package io.github.lukaszgryglicki.chess;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Insets;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.text.InputType;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The start screen (N, "CPU move", "Demo") and the game screen. The game,
 * its rules and the way pieces are picked, dropped and let go are chess.go's:
 * this only shows what it reports and passes the finger on.
 */
public final class MainActivity extends Activity implements Engine.Listener {
    private static final String TAG = "chess";
    private static final int BACK = 0xFF262421, TEXT = 0xFFECECEC, DIM = 0xFFA8A8A8, WARN = 0xFFFF8A65;
    private static final int RESULT = 0xFFFFD54F;
    private static final int MATCH = ViewGroup.LayoutParams.MATCH_PARENT, WRAP = ViewGroup.LayoutParams.WRAP_CONTENT;

    private SharedPreferences prefs; // the game in progress: settings and moves
    private FrameLayout root;
    private ScrollView startView;
    private EditText limitEdit;
    private CheckBox cpuBox, demoBox;
    private TextView startError;
    private LinearLayout gameView, panel;
    private BoardView board;
    private ProgressBar busy;
    private TextView status, info;
    private Button cpuButton;

    private Engine engine;
    private boolean visible, playing, over;
    private int failures; // times the game process was lost since the last move

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences("game", MODE_PRIVATE);
        root = new FrameLayout(this);
        root.setBackgroundColor(BACK);
        root.setOnApplyWindowInsetsListener((v, in) -> { // keep clear of the system bars
            if (Build.VERSION.SDK_INT >= 30) {
                Insets i = in.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout()
                        | WindowInsets.Type.ime());
                v.setPadding(i.left, i.top, i.right, i.bottom);
            } else {
                v.setPadding(in.getSystemWindowInsetLeft(), in.getSystemWindowInsetTop(),
                        in.getSystemWindowInsetRight(), in.getSystemWindowInsetBottom());
            }
            return in;
        });
        setContentView(root);
        buildStart();
        buildGame();
        testStart(getIntent());
        if (prefs.getBoolean("active", false)) {
            showGame();
        } else {
            showStart(null);
        }
    }

    private int dp(float v) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()));
    }

    private TextView text(String s, float sp, int color) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        return t;
    }

    private static LinearLayout.LayoutParams below(int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(MATCH, WRAP);
        p.topMargin = top;
        return p;
    }

    private void buildStart() {
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(dp(24), dp(24), dp(24), dp(24));
        TextView title = text("Chess", 30, TEXT);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        col.addView(title);
        col.addView(text("N: the computer's limit for one move", 17, TEXT), below(dp(24)));
        limitEdit = new EditText(this);
        limitEdit.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED
                | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        limitEdit.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24);
        limitEdit.setSingleLine();
        limitEdit.setSelectAllOnFocus(true);
        limitEdit.setImeOptions(EditorInfo.IME_ACTION_GO);
        limitEdit.setOnEditorActionListener((v, action, ev) -> {
            if (ev == null || ev.getAction() == KeyEvent.ACTION_DOWN) {
                start();
            }
            return true;
        });
        col.addView(limitEdit, below(dp(4)));
        col.addView(text("30: at most 30 seconds a move\n0: no time limit, it moves when its memory is used up\n"
                + "-7: search depth 7", 14, DIM), below(dp(4)));
        cpuBox = new CheckBox(this);
        cpuBox.setText("CPU MOVE button: the computer may move for you");
        cpuBox.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        col.addView(cpuBox, below(dp(24)));
        demoBox = new CheckBox(this);
        demoBox.setText("Demo: the computer plays both sides");
        demoBox.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        col.addView(demoBox, below(dp(8)));
        Button go = new Button(this);
        go.setText("Start");
        go.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        go.setMinHeight(dp(56));
        go.setOnClickListener(v -> start());
        col.addView(go, below(dp(24)));
        startError = text("", 15, WARN);
        col.addView(startError, below(dp(12)));
        startView = new ScrollView(this);
        startView.setFillViewport(true);
        startView.addView(col, new ViewGroup.LayoutParams(MATCH, WRAP));
    }

    private void buildGame() {
        board = new BoardView(this);
        board.setPictures(pictures());
        board.setWake(() -> { // after "Touch the board to go on."
            if (playing && engine == null) {
                failures = 0;
                startEngine();
            }
        });
        busy = new ProgressBar(this);
        status = text("", 17, TEXT);
        status.setMinLines(2);
        LinearLayout line = new LinearLayout(this);
        line.setGravity(Gravity.CENTER_VERTICAL);
        line.addView(status, new LinearLayout.LayoutParams(0, WRAP, 1));
        line.addView(busy, new LinearLayout.LayoutParams(dp(28), dp(28)));
        cpuButton = new Button(this);
        cpuButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        cpuButton.setMinHeight(dp(56));
        cpuButton.setOnClickListener(v -> {
            if (engine != null) {
                engine.send("cpu");
            }
        });
        Button fresh = new Button(this);
        fresh.setText("New game");
        fresh.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        fresh.setMinHeight(dp(56));
        fresh.setOnClickListener(v -> askNewGame());
        info = text("", 12, DIM);
        panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(12), dp(8), dp(12), dp(8));
        panel.addView(line, below(0));
        panel.addView(cpuButton, below(dp(8)));
        panel.addView(new View(this), new LinearLayout.LayoutParams(MATCH, 0, 1));
        panel.addView(info, below(dp(8)));
        panel.addView(fresh, below(dp(4)));
        gameView = new LinearLayout(this);
    }

    /**
     * Where the game draws the piece pictures: a directory of this version of
     * the app, so that a new version draws its own. Older ones are removed.
     */
    private File pictures() {
        long version = 0;
        try {
            version = getPackageManager().getPackageInfo(getPackageName(), 0).lastUpdateTime;
        } catch (PackageManager.NameNotFoundException e) {
            // not to be: it is this app
        }
        File dir = new File(getCacheDir(), "pictures-" + version);
        File[] all = getCacheDir().listFiles();
        for (File old : all == null ? new File[0] : all) {
            if (old.getName().startsWith("pictures-") && !old.equals(dir)) {
                File[] sizes = old.listFiles();
                for (File size : sizes == null ? new File[0] : sizes) {
                    File[] files = size.listFiles();
                    for (File f : files == null ? new File[0] : files) {
                        f.delete();
                    }
                    size.delete();
                }
                old.delete();
            }
        }
        return dir;
    }

    /** The board takes the short side of the screen, the rest goes beside or below it. */
    private void layoutGame() {
        boolean wide = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        gameView.removeAllViews();
        gameView.setOrientation(wide ? LinearLayout.HORIZONTAL : LinearLayout.VERTICAL);
        gameView.addView(board, new LinearLayout.LayoutParams(wide ? WRAP : MATCH, wide ? MATCH : WRAP));
        gameView.addView(panel, new LinearLayout.LayoutParams(wide ? 0 : MATCH, wide ? MATCH : 0, 1));
    }

    @Override
    public void onConfigurationChanged(Configuration c) {
        super.onConfigurationChanged(c);
        layoutGame();
    }

    private void showStart(String error) {
        playing = false;
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        limitEdit.setText(prefs.getString("limit", "30"));
        cpuBox.setChecked(false);
        demoBox.setChecked(false);
        startError.setText(error == null ? "" : error);
        root.removeAllViews();
        root.addView(startView, new FrameLayout.LayoutParams(MATCH, MATCH));
    }

    private void start() {
        String n = limitEdit.getText().toString().trim();
        startError.setText("");
        if (n.isEmpty()) {
            startError.setText("Type a number for N.");
            return;
        }
        ((InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE))
                .hideSoftInputFromWindow(limitEdit.getWindowToken(), 0);
        prefs.edit().putString("limit", n).putBoolean("cpu", cpuBox.isChecked())
                .putBoolean("demo", demoBox.isChecked()).putString("moves", "").putString("fen", "")
                .putBoolean("active", true).apply();
        showGame();
    }

    /**
     * Testing aid of debuggable builds (DEBUG=1 sh build.sh), for "adb shell am start -S -n .../.MainActivity":
     * --es limit -3 [--ez cpu true] [--ez demo true] [--es fen ...] [--es moves ...] starts that game at once,
     * --ez wide true|false turns the screen, --ei die 3000 takes the game process away every 3 seconds.
     */
    private void testStart(Intent in) {
        if ((getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) == 0 || in == null) {
            return;
        }
        if (in.hasExtra("wide")) {
            setRequestedOrientation(in.getBooleanExtra("wide", false) ? ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                    : ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        }
        final int die = in.getIntExtra("die", 0);
        if (die > 0) {
            root.postDelayed(new Runnable() {
                @Override
                public void run() {
                    if (engine != null) {
                        engine.kill();
                    }
                    root.postDelayed(this, die);
                }
            }, die);
        }
        if (in.hasExtra("limit")) {
            String fen = in.getStringExtra("fen"), moves = in.getStringExtra("moves");
            prefs.edit().putString("limit", in.getStringExtra("limit")).putBoolean("cpu", in.getBooleanExtra("cpu", false))
                    .putBoolean("demo", in.getBooleanExtra("demo", false)).putString("moves", moves == null ? "" : moves)
                    .putString("fen", fen == null ? "" : fen).putBoolean("active", true).apply();
        }
    }

    private void showGame() {
        playing = true;
        failures = 0;
        root.removeAllViews();
        root.addView(gameView, new FrameLayout.LayoutParams(MATCH, MATCH));
        layoutGame();
        startEngine();
    }

    /** Starts the game process from the record: the settings and the moves played so far. */
    private void startEngine() {
        stopEngine();
        over = false;
        status.setText("Starting...");
        busy.setVisibility(View.VISIBLE);
        cpuButton.setVisibility(View.GONE);
        List<String> args = new ArrayList<>();
        args.add(prefs.getString("limit", "30"));
        args.add(prefs.getBoolean("cpu", false) ? "1" : "0");
        args.add(prefs.getBoolean("demo", false) ? "1" : "0");
        for (String m : prefs.getString("moves", "").split(" ")) {
            if (!m.isEmpty()) {
                args.add(m);
            }
        }
        try {
            engine = new Engine(this, args, prefs.getString("fen", ""), this);
            if (!visible) {
                engine.pause();
            }
        } catch (IOException e) {
            Log.e(TAG, "cannot start the game", e);
            prefs.edit().putBoolean("active", false).apply();
            showStart("The game cannot be started: " + e.getMessage());
        }
    }

    private void stopEngine() {
        if (engine != null) {
            Engine e = engine;
            engine = null;
            e.kill();
        }
        board.detach(!prefs.getBoolean("active", false));
    }

    @Override
    public void onLine(Engine e, String line) {
        if (e != engine) {
            return; // a game that has been left
        }
        int sp = line.indexOf(' ');
        String word = sp < 0 ? line : line.substring(0, sp), rest = sp < 0 ? "" : line.substring(sp + 1);
        try {
            switch (word) {
                case "ready": { // pid, units to a square, threads, hash MB
                    String[] f = rest.split(" ");
                    e.setPid(Integer.parseInt(f[0]));
                    board.attach(e, Integer.parseInt(f[1]));
                    info.setText("N = " + prefs.getString("limit", "") + ", " + f[2] + " threads, " + f[3] + " MB");
                    break;
                }
                case "sprites": { // size, directory
                    String[] f = rest.split(" ", 2);
                    board.sprites(Integer.parseInt(f[0]), f[1]);
                    break;
                }
                case "state":
                    state(rest.split(" ", 11));
                    break;
                case "moves":
                    if (!rest.equals(prefs.getString("moves", ""))) {
                        prefs.edit().putString("moves", rest).apply();
                        failures = 0; // the game gets on
                    }
                    break;
                case "error":
                    prefs.edit().putBoolean("active", false).apply();
                    stopEngine();
                    showStart(rest.startsWith("bad limit")
                            ? rest + ": N is seconds (30, 2.5), 0, or a whole negative number for a depth (-7)." : rest);
                    break;
                default:
                    Log.i(TAG, line);
                    break;
            }
        } catch (RuntimeException ex) {
            Log.e(TAG, "bad line from the game: " + line, ex);
        }
    }

    /** board, side, selected, last from, last to, promotion, held, thinking, over, button, status. */
    private void state(String[] f) {
        board.state(f[0], Integer.parseInt(f[2]), Integer.parseInt(f[3]), Integer.parseInt(f[4]),
                Integer.parseInt(f[5]), Integer.parseInt(f[6]));
        boolean thinking = f[7].equals("1");
        over = f[8].equals("1");
        String said = f.length > 10 ? f[10] : "";
        status.setText(over ? result(said) : said);
        status.setTextColor(over ? RESULT : TEXT);
        busy.setVisibility(thinking ? View.VISIBLE : View.INVISIBLE);
        cpuButton.setVisibility(f[9].equals("1") ? View.VISIBLE : View.GONE);
        cpuButton.setEnabled(!thinking && !over);
        cpuButton.setText(thinking ? "Thinking..." : "CPU move (plays for you)");
        if (thinking) { // the screen must not go dark, and stop the game, while it thinks
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }
    }

    /** The game's last word, "1-0 checkmate" or "1/2-1/2 material", said for a player. */
    private static String result(String s) {
        String why = s.substring(s.indexOf(' ') + 1);
        if (why.equals("material")) {
            why = "not enough material";
        } else if (why.equals("fifty-move")) {
            why = "fifty moves without a capture or pawn move";
        }
        return s.startsWith("1-0 ") ? "White wins: " + why : s.startsWith("0-1 ") ? "Black wins: " + why
                : s.startsWith("1/2-1/2 ") ? "Draw: " + why : s;
    }

    @Override
    public void onExit(Engine e) {
        if (e != engine) {
            return;
        }
        // the system may take the process away at any time: go on from the record,
        // at once while the game is watched, else when the app is looked at again
        engine = null;
        board.detach(false);
        if (!visible) {
            return;
        }
        if (++failures >= 3) { // it keeps going away: no more until the player asks
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            status.setText("The game stopped. Touch the board to go on.");
            busy.setVisibility(View.INVISIBLE);
            cpuButton.setEnabled(false);
            return;
        }
        Log.w(TAG, "the game process is gone: starting it again");
        startEngine();
    }

    private void askNewGame() {
        if (over || prefs.getString("moves", "").isEmpty()) {
            newGame();
            return;
        }
        new AlertDialog.Builder(this).setMessage("Leave this game and start a new one?")
                .setPositiveButton("New game", (d, w) -> newGame()).setNegativeButton("Keep playing", null).show();
    }

    private void newGame() {
        prefs.edit().putBoolean("active", false).apply();
        stopEngine();
        showStart(null);
    }

    @Override
    protected void onStart() {
        super.onStart();
        visible = true;
        if (engine != null) {
            engine.resume();
        } else if (playing) {
            failures = 0;
            startEngine();
        }
    }

    @Override
    protected void onStop() {
        visible = false;
        if (engine != null) {
            engine.pause();
        }
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        stopEngine();
        super.onDestroy();
    }
}
