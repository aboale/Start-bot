"""
Alpha Extraction — Android App (Kivy) — Full Version
======================================================
Adds on top of the base pipeline (Eq. 1-12):
  - Technical indicators as extra sub-signals: RSI, MACD, Bollinger %B, Volume z-score
  - True cross-sectional mode: compare several coins against each other
  - Binance Futures context: funding rate, open interest, order-book imbalance
  - Crypto Fear & Greed index (market-wide sentiment)
  - Simple historical backtest diagnostic (hit rate of signal sign vs forward return)
  - ATR-based risk info (suggested stop distance) — informational only
  - Local price alert (checked when you tap the analyze button, no background service)

Build:
    pip install buildozer cython
    buildozer -v android debug

Desktop preview:
    pip install kivy requests numpy
    python main.py

DISCLAIMER: this app produces statistical/technical readings for
educational and analytical purposes. It is not financial advice, and
nothing here should be treated as a guaranteed trading signal.
"""

import threading
import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.clock import Clock
from kivy.graphics import Color, Line, Rectangle
from kivy.storage.jsonstore import JsonStore

from alpha_core import (
    AlphaExtractionBot, build_signal_matrix, build_cross_sectional_matrix,
    simple_backtest, DEFAULT_WINDOWS,
)
from data_providers_lite import (
    fetch_with_fallback, PROVIDERS, fetch_binance_ohlcv,
    fetch_binance_funding_rate, fetch_binance_open_interest,
    fetch_binance_order_book_imbalance, fetch_fear_greed_index,
)

STORE_FILE = "alpha_app_settings.json"


# ---------------------------------------------------------------------------
# Reusable price chart (Kivy canvas, no matplotlib)
# ---------------------------------------------------------------------------

class PriceChartWidget(RelativeLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prices = []
        self.min_label = Label(text="", size_hint=(None, None), size=(80, 20),
                                font_size=12, color=(0.6, 0.6, 0.6, 1))
        self.max_label = Label(text="", size_hint=(None, None), size=(80, 20),
                                font_size=12, color=(0.6, 0.6, 0.6, 1))
        self.add_widget(self.min_label)
        self.add_widget(self.max_label)
        self.bind(size=self._redraw, pos=self._redraw)

    def plot(self, prices):
        self.prices = list(prices)
        self._redraw()

    def _redraw(self, *args):
        self.canvas.before.clear()
        if not self.prices or self.width <= 1 or self.height <= 1:
            return

        pad_left, pad_right, pad_top, pad_bottom = 10, 10, 15, 25
        chart_w = self.width - pad_left - pad_right
        chart_h = self.height - pad_top - pad_bottom
        if chart_w <= 0 or chart_h <= 0:
            return

        lo, hi = min(self.prices), max(self.prices)
        span = (hi - lo) or 1.0
        n = len(self.prices)

        points = []
        for i, p in enumerate(self.prices):
            x = self.x + pad_left + (i / max(n - 1, 1)) * chart_w
            y = self.y + pad_bottom + ((p - lo) / span) * chart_h
            points.extend([x, y])

        with self.canvas.before:
            Color(0.12, 0.12, 0.14, 1)
            Rectangle(pos=self.pos, size=self.size)
            Color(0.3, 0.3, 0.33, 1)
            Line(points=[self.x + pad_left, self.y + pad_bottom,
                         self.x + pad_left + chart_w, self.y + pad_bottom], width=1)
            up = self.prices[-1] >= self.prices[0]
            Color(0.2, 0.8, 0.4, 1) if up else Color(0.9, 0.3, 0.3, 1)
            Line(points=points, width=1.5)

        self.max_label.text = f"{hi:.4g}"
        self.max_label.pos = (self.x + pad_left, self.y + self.height - pad_top - 5)
        self.min_label.text = f"{lo:.4g}"
        self.min_label.pos = (self.x + pad_left, self.y + 2)


# ---------------------------------------------------------------------------
# Settings popup — API keys + price alert threshold
# ---------------------------------------------------------------------------

class SettingsPopup(Popup):
    def __init__(self, store, **kwargs):
        super().__init__(**kwargs)
        self.store = store
        self.title = "إعدادات مفاتيح API والتنبيهات"
        self.size_hint = (0.9, 0.8)

        layout = GridLayout(cols=2, spacing=10, padding=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter("height"))

        layout.add_widget(Label(text="[b]مفاتيح API (اختياري)[/b]", markup=True,
                                 size_hint_y=None, height=30))
        layout.add_widget(Label(text="", size_hint_y=None, height=30))

        self.inputs = {}
        for name in PROVIDERS.keys():
            if name == "Binance":
                continue
            layout.add_widget(Label(text=name, size_hint_y=None, height=40))
            existing = ""
            if self.store.exists("api_keys") and name in self.store.get("api_keys"):
                existing = self.store.get("api_keys")[name]
            ti = TextInput(text=existing, multiline=False, size_hint_y=None, height=40,
                            password=True)
            self.inputs[name] = ti
            layout.add_widget(ti)

        layout.add_widget(Label(text="[b]تنبيه سعر (يُتحقق منه عند كل تحليل)[/b]", markup=True,
                                 size_hint_y=None, height=30))
        layout.add_widget(Label(text="", size_hint_y=None, height=30))

        layout.add_widget(Label(text="عتبة الإشارة المُدمجة", size_hint_y=None, height=40))
        existing_threshold = ""
        if self.store.exists("alert"):
            v = self.store.get("alert").get("threshold")
            existing_threshold = "" if v is None else str(v)
        self.alert_input = TextInput(text=existing_threshold, multiline=False,
                                      size_hint_y=None, height=40,
                                      input_filter="float")
        layout.add_widget(self.alert_input)

        scroll = ScrollView(size_hint=(1, 0.8))
        scroll.add_widget(layout)

        root = BoxLayout(orientation="vertical")
        root.add_widget(scroll)

        save_btn = Button(text="حفظ", size_hint=(1, 0.2))
        save_btn.bind(on_release=self.save)
        root.add_widget(save_btn)

        self.content = root

    def save(self, *args):
        keys = {name: ti.text.strip() for name, ti in self.inputs.items()}
        self.store.put("api_keys", **keys)
        try:
            threshold = float(self.alert_input.text) if self.alert_input.text else None
        except ValueError:
            threshold = None
        self.store.put("alert", threshold=threshold)
        self.dismiss()


# ---------------------------------------------------------------------------
# Tab 1: single-asset analysis (momentum + indicators + futures + sentiment)
# ---------------------------------------------------------------------------

class SingleAssetTab(BoxLayout):
    def __init__(self, store, **kwargs):
        super().__init__(orientation="vertical", padding=10, spacing=8, **kwargs)
        self.store = store

        row = BoxLayout(size_hint_y=None, height=45, spacing=10)
        row.add_widget(Label(text="الرمز:", size_hint_x=0.25))
        self.symbol_input = TextInput(text="BTCUSDT", multiline=False)
        row.add_widget(self.symbol_input)
        self.add_widget(row)

        hint = Label(text="عملات باينانس بدون شرطة: BTCUSDT, ETHUSDT...",
                     size_hint_y=None, height=22, font_size=12, color=(0.6, 0.6, 0.6, 1))
        self.add_widget(hint)

        row2 = BoxLayout(size_hint_y=None, height=45, spacing=10)
        row2.add_widget(Label(text="المصدر:", size_hint_x=0.25))
        self.source_spinner = Spinner(
            text="Binance (بدون مفتاح)",
            values=["تلقائي (كل المصادر)", "Binance (بدون مفتاح)"] +
                   [n for n in PROVIDERS.keys() if n != "Binance"],
        )
        row2.add_widget(self.source_spinner)
        self.add_widget(row2)

        row3 = BoxLayout(size_hint_y=None, height=45, spacing=10)
        row3.add_widget(Label(text="المدة:", size_hint_x=0.25))
        self.period_spinner = Spinner(text="1y", values=["1mo", "3mo", "6mo", "1y", "2y"])
        row3.add_widget(self.period_spinner)
        self.add_widget(row3)

        btn_row = BoxLayout(size_hint_y=None, height=45, spacing=10)
        run_btn = Button(text="تحليل")
        run_btn.bind(on_release=self.run_pipeline)
        backtest_btn = Button(text="اختبار تاريخي")
        backtest_btn.bind(on_release=self.run_backtest)
        btn_row.add_widget(run_btn)
        btn_row.add_widget(backtest_btn)
        self.add_widget(btn_row)

        self.status_label = Label(text="", size_hint_y=None, height=26)
        self.add_widget(self.status_label)

        self.chart = PriceChartWidget(size_hint_y=None, height=160)
        self.add_widget(self.chart)

        self.result_label = Label(text="", halign="right", valign="top", markup=True)
        self.result_label.bind(size=self._update_text_size)
        scroll = ScrollView()
        scroll.add_widget(self.result_label)
        self.add_widget(scroll)

        self._last_prices = None

    def _update_text_size(self, instance, size):
        instance.text_size = (size[0], None)

    def _get_source_order(self):
        selected = self.source_spinner.text
        if selected.startswith("تلقائي"):
            return None
        if selected.startswith("Binance"):
            return ["Binance"]
        return [selected]

    def run_pipeline(self, *args):
        self.status_label.text = "جاري الجلب والتحليل..."
        self.result_label.text = ""
        threading.Thread(target=self._run_pipeline_thread, daemon=True).start()

    def _run_pipeline_thread(self):
        symbol = self.symbol_input.text.strip()
        period = self.period_spinner.text
        order = self._get_source_order()
        using_binance = (order == ["Binance"]) or (order is None)

        try:
            api_keys = self.store.get("api_keys") if self.store.exists("api_keys") else {}

            highs = lows = volumes = None
            used_source = None
            if using_binance:
                try:
                    ohlcv = fetch_binance_ohlcv(symbol, period=period, interval="1d")
                    prices_list = ohlcv["close"]
                    highs, lows, volumes = ohlcv["high"], ohlcv["low"], ohlcv["volume"]
                    used_source = "Binance"
                except Exception:
                    prices_list, used_source = fetch_with_fallback(
                        symbol, period=period, interval="1d", api_keys=api_keys, order=order
                    )
            else:
                prices_list, used_source = fetch_with_fallback(
                    symbol, period=period, interval="1d", api_keys=api_keys, order=order
                )

            prices = np.array(prices_list, dtype=float)

            R, S_current, labels, extras = build_signal_matrix(
                prices, windows=DEFAULT_WINDOWS,
                highs=np.array(highs) if highs else None,
                lows=np.array(lows) if lows else None,
                volumes=np.array(volumes) if volumes else None,
            )
            bot = AlphaExtractionBot(R, d=min(20, R.shape[1] - 1))
            result = bot.run(S_current)

            lines = [f"[b]المصدر:[/b] {used_source}", ""]
            lines.append(f"{'الإشارة':<18}{'sigma':>8}{'E_norm':>9}{'الوزن':>8}")
            for lbl, sig, en, w in zip(labels, result["sigma"], result["E_norm"], result["weights"]):
                lines.append(f"{lbl:<18}{sig:>8.4f}{en:>9.4f}{w:>8.4f}")
            lines.append("")
            lines.append(f"[b]الإشارة المُدمجة:[/b] {result['combined_signal']:.4f}")

            if "atr_last" in extras:
                atr_val = extras["atr_last"]
                last_price = prices[-1]
                lines.append("")
                lines.append("[b]--- إدارة المخاطر (معلوماتي) ---[/b]")
                lines.append(f"ATR الحالي: {atr_val:.4g}")
                lines.append(f"وقف خسارة مقترح (1.5xATR): {last_price - 1.5*atr_val:.4g} / {last_price + 1.5*atr_val:.4g}")
                lines.append(f"هدف ربح مقترح (2xATR): {last_price - 2*atr_val:.4g} / {last_price + 2*atr_val:.4g}")

            if using_binance:
                lines.append("")
                lines.append("[b]--- سياق إضافي (باينانس فيوتشرز + معنويات) ---[/b]")
                try:
                    fr = fetch_binance_funding_rate(symbol)
                    lines.append(f"معدل التمويل: {fr['funding_rate']*100:.4f}%")
                except Exception as e:
                    lines.append(f"معدل التمويل: غير متاح ({e})")

                try:
                    oi = fetch_binance_open_interest(symbol)
                    lines.append(f"الفائدة المفتوحة: {oi['open_interest']:.2f}")
                except Exception as e:
                    lines.append(f"الفائدة المفتوحة: غير متاحة ({e})")

                try:
                    ob = fetch_binance_order_book_imbalance(symbol)
                    direction = "ضغط شراء" if ob['imbalance'] > 0 else "ضغط بيع"
                    lines.append(f"توازن دفتر الأوامر: {ob['imbalance']:+.3f} ({direction})")
                except Exception as e:
                    lines.append(f"دفتر الأوامر: غير متاح ({e})")

            try:
                fg = fetch_fear_greed_index()
                lines.append(f"مؤشر الخوف والطمع: {fg['value']} — {fg['classification']}")
            except Exception as e:
                lines.append(f"مؤشر الخوف والطمع: غير متاح ({e})")

            alert_cfg = self.store.get("alert") if self.store.exists("alert") else {}
            threshold = alert_cfg.get("threshold")
            if threshold is not None:
                triggered = abs(result["combined_signal"]) >= abs(threshold)
                lines.append("")
                if triggered:
                    lines.append(f"[color=ffcc00][b]تنبيه: الإشارة تجاوزت العتبة ({threshold})[/b][/color]")
                else:
                    lines.append(f"(لم تتجاوز الإشارة عتبة التنبيه المحفوظة: {threshold})")

            text = "\n".join(lines)
            status = "تم بنجاح"
            chart_prices = prices[-180:]
            self._last_prices = prices
        except Exception as e:
            text = f"[color=ff3333]خطأ: {e}[/color]"
            status = "فشل"
            chart_prices = None

        Clock.schedule_once(lambda dt: self._update_ui(status, text, chart_prices))

    def _update_ui(self, status, text, chart_prices=None):
        self.status_label.text = status
        self.result_label.text = text
        if chart_prices is not None:
            self.chart.plot(chart_prices)

    def run_backtest(self, *args):
        if self._last_prices is None:
            self.status_label.text = "شغّل التحليل أولاً قبل الاختبار التاريخي"
            return
        self.status_label.text = "جاري الاختبار التاريخي..."
        threading.Thread(target=self._run_backtest_thread, daemon=True).start()

    def _run_backtest_thread(self):
        try:
            bt = simple_backtest(self._last_prices, lookback_bars=250, rebalance_every=5)
            lines = [
                "[b]--- نتائج الاختبار التاريخي (تشخيصي فقط) ---[/b]",
                f"عدد العينات: {bt['n_samples']}",
                f"نسبة الإصابة: {bt['hit_rate']*100:.1f}%",
            ]
            if bt["avg_forward_return_when_long"] is not None:
                lines.append(f"متوسط العائد اللاحق عند إشارة شراء: {bt['avg_forward_return_when_long']*100:.3f}%")
            if bt["avg_forward_return_when_short"] is not None:
                lines.append(f"متوسط العائد اللاحق عند إشارة بيع: {bt['avg_forward_return_when_short']*100:.3f}%")
            lines.append("")
            lines.append("[i]تشخيص وصفي على بيانات تاريخية، وليس ضمانًا لأداء مستقبلي.[/i]")
            text = "\n".join(lines)
            status = "اكتمل الاختبار التاريخي"
        except Exception as e:
            text = f"[color=ff3333]خطأ في الاختبار التاريخي: {e}[/color]"
            status = "فشل"

        Clock.schedule_once(lambda dt: self._update_ui(status, text, None))


# ---------------------------------------------------------------------------
# Tab 2: true cross-sectional mode — compare multiple coins
# ---------------------------------------------------------------------------

class CrossSectionalTab(BoxLayout):
    def __init__(self, store, **kwargs):
        super().__init__(orientation="vertical", padding=10, spacing=8, **kwargs)
        self.store = store

        info = Label(
            text="أدخل عدة رموز مفصولة بفاصلة للمقارنة المقطعية الحقيقية",
            size_hint_y=None, height=40, font_size=12, halign="right", valign="middle",
        )
        info.bind(size=lambda i, s: setattr(i, "text_size", s))
        self.add_widget(info)

        row = BoxLayout(size_hint_y=None, height=45, spacing=10)
        row.add_widget(Label(text="الرموز:", size_hint_x=0.25))
        self.symbols_input = TextInput(text="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT", multiline=False)
        row.add_widget(self.symbols_input)
        self.add_widget(row)

        row2 = BoxLayout(size_hint_y=None, height=45, spacing=10)
        row2.add_widget(Label(text="المدة:", size_hint_x=0.25))
        self.period_spinner = Spinner(text="6mo", values=["1mo", "3mo", "6mo", "1y", "2y"])
        row2.add_widget(self.period_spinner)
        self.add_widget(row2)

        run_btn = Button(text="قارن الأصول", size_hint_y=None, height=45)
        run_btn.bind(on_release=self.run_pipeline)
        self.add_widget(run_btn)

        self.status_label = Label(text="", size_hint_y=None, height=26)
        self.add_widget(self.status_label)

        self.result_label = Label(text="", halign="right", valign="top", markup=True)
        self.result_label.bind(size=lambda i, s: setattr(i, "text_size", (s[0], None)))
        scroll = ScrollView()
        scroll.add_widget(self.result_label)
        self.add_widget(scroll)

    def run_pipeline(self, *args):
        self.status_label.text = "جاري الجلب والمقارنة..."
        self.result_label.text = ""
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        try:
            symbols = [s.strip().upper() for s in self.symbols_input.text.split(",") if s.strip()]
            if len(symbols) < 2:
                raise ValueError("أدخل رمزين على الأقل للمقارنة المقطعية")

            period = self.period_spinner.text
            price_dict = {}
            for sym in symbols:
                ohlcv = fetch_binance_ohlcv(sym, period=period, interval="1d")
                price_dict[sym] = ohlcv["close"]

            R, S_current, names = build_cross_sectional_matrix(price_dict)
            bot = AlphaExtractionBot(R, d=min(20, R.shape[1] - 1))
            result = bot.run(S_current)

            lines = ["[b]نتائج المقارنة المقطعية:[/b]", ""]
            lines.append(f"{'الرمز':<12}{'sigma':>8}{'E_norm':>9}{'الوزن':>9}")
            for name, sig, en, w in zip(names, result["sigma"], result["E_norm"], result["weights"]):
                lines.append(f"{name:<12}{sig:>8.4f}{en:>9.4f}{w:>9.4f}")

            lines.append("")
            lines.append(f"[b]الإشارة المُدمجة الكلية:[/b] {result['combined_signal']:.6f}")
            lines.append("")

            ranked = sorted(zip(names, result["weights"]), key=lambda x: -x[1])
            lines.append("[b]الترتيب حسب الوزن:[/b]")
            for name, w in ranked:
                direction = "قوة نسبية إيجابية" if w > 0 else "ضعف نسبي"
                lines.append(f"  {name}: {w:+.4f} ({direction})")

            text = "\n".join(lines)
            status = "تم بنجاح"
        except Exception as e:
            text = f"[color=ff3333]خطأ: {e}[/color]"
            status = "فشل"

        Clock.schedule_once(lambda dt: self._update_ui(status, text))

    def _update_ui(self, status, text):
        self.status_label.text = status
        self.result_label.text = text


# ---------------------------------------------------------------------------
# Root app
# ---------------------------------------------------------------------------

class RootLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.store = JsonStore(STORE_FILE)

        top_bar = BoxLayout(size_hint_y=None, height=45, padding=(10, 5))
        top_bar.add_widget(Label(text="[b]بوت استخراج الألفا[/b]", markup=True, font_size=18))
        settings_btn = Button(text="اعدادات", size_hint_x=0.3)
        settings_btn.bind(on_release=self.open_settings)
        top_bar.add_widget(settings_btn)
        self.add_widget(top_bar)

        tabs = TabbedPanel(do_default_tab=False)

        tab1 = TabbedPanelItem(text="أصل واحد + مؤشرات")
        tab1.add_widget(SingleAssetTab(self.store))
        tabs.add_widget(tab1)

        tab2 = TabbedPanelItem(text="مقارنة عملات")
        tab2.add_widget(CrossSectionalTab(self.store))
        tabs.add_widget(tab2)

        tabs.default_tab = tab1
        self.add_widget(tabs)

        disclaimer = Label(
            text="هذه أداة تحليل إحصائي وليست نصيحة مالية",
            size_hint_y=None, height=24, font_size=11, color=(0.6, 0.6, 0.6, 1),
        )
        self.add_widget(disclaimer)

    def open_settings(self, *args):
        SettingsPopup(self.store).open()


class AlphaExtractionApp(App):
    def build(self):
        self.title = "Alpha Extraction Bot"
        return RootLayout()


if __name__ == "__main__":
    AlphaExtractionApp().run()
