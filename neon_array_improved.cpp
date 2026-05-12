#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>

#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define HAVE_NEON 1
#else
    #define HAVE_NEON 0
#endif

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum += -val;
    }
    return sum;
}

int64_t process_array_neon(const int32_t* data, size_t n) {
#if HAVE_NEON
    int64_t sum = 0;
    int32x4_t acc = vdupq_n_s32(0);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t vec = vld1q_s32(data + i);

        uint32x4_t mask_pos = vcgtq_s32(vec, vdupq_n_s32(0));
        uint32x4_t mask_neg = vcltq_s32(vec, vdupq_n_s32(0));

        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);

        int32x4_t pos_part = vandq_s32(vec, vreinterpretq_s32_u32(mask_pos));
        int32x4_t neg_part = vandq_s32(abs_val, vreinterpretq_s32_u32(mask_neg));
        int32x4_t contrib = vorrq_s32(pos_part, neg_part);

        acc = vaddq_s32(acc, contrib);
    }

    #if defined(__aarch64__)
        sum += vaddvq_s32(acc);
    #else
        int32x2_t sum_pairs = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        int32x2_t sum_final = vpadd_s32(sum_pairs, sum_pairs);
        sum += vget_lane_s32(sum_final, 0);
    #endif

    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum += -val;
    }
    return sum;
#else
    return process_array_scalar(data, n);
#endif
}

int64_t process_array_neon_unrolled(const int32_t* data, size_t n) {
#if HAVE_NEON
    int64_t sum = 0;

    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        int32x4_t vec1 = vld1q_s32(data + i);
        int32x4_t vec2 = vld1q_s32(data + i + 4);

        uint32x4_t mask_pos1 = vcgtq_s32(vec1, vdupq_n_s32(0));
        uint32x4_t mask_neg1 = vcltq_s32(vec1, vdupq_n_s32(0));

        int32x4_t sign1 = vshrq_n_s32(vec1, 31);
        int32x4_t abs_val1 = vsubq_s32(veorq_s32(vec1, sign1), sign1);

        int32x4_t pos_part1 = vandq_s32(vec1, vreinterpretq_s32_u32(mask_pos1));
        int32x4_t neg_part1 = vandq_s32(abs_val1, vreinterpretq_s32_u32(mask_neg1));
        int32x4_t contrib1 = vorrq_s32(pos_part1, neg_part1);

        acc1 = vaddq_s32(acc1, contrib1);

        uint32x4_t mask_pos2 = vcgtq_s32(vec2, vdupq_n_s32(0));
        uint32x4_t mask_neg2 = vcltq_s32(vec2, vdupq_n_s32(0));

        int32x4_t sign2 = vshrq_n_s32(vec2, 31);
        int32x4_t abs_val2 = vsubq_s32(veorq_s32(vec2, sign2), sign2);

        int32x4_t pos_part2 = vandq_s32(vec2, vreinterpretq_s32_u32(mask_pos2));
        int32x4_t neg_part2 = vandq_s32(abs_val2, vreinterpretq_s32_u32(mask_neg2));
        int32x4_t contrib2 = vorrq_s32(pos_part2, neg_part2);

        acc2 = vaddq_s32(acc2, contrib2);
    }

    int32x4_t acc = vaddq_s32(acc1, acc2);

    #if defined(__aarch64__)
        sum += vaddvq_s32(acc);
    #else
        int32x2_t sum_pairs = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        int32x2_t sum_final = vpadd_s32(sum_pairs, sum_pairs);
        sum += vget_lane_s32(sum_final, 0);
    #endif

    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum += -val;
    }
    return sum;
#else
    return process_array_scalar(data, n);
#endif
}

struct BenchmarkResult {
    size_t size;
    double scalar_time;
    double neon_time;
    double unrolled_time;
    double speedup_neon;
    double speedup_unrolled;
    int64_t result_value;
};

void generate_html_chart(const std::vector<BenchmarkResult>& results) {
    std::string path = "neon_benchmark_chart.html";
    std::ofstream html(path);

    if (!html.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл для записи: " << path << "\n";
        return;
    }

    html << R"(<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARM NEON - Анализ производительности</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0c14;
            min-height: 100vh;
            padding: 28px;
            color: #d2d7e8;
        }

        .page {
            max-width: 1300px;
            margin: 0 auto;
        }

        .header {
            margin-bottom: 24px;
        }

        .header h1 {
            font-size: 1.6em;
            font-weight: 600;
            color: #7b9ef0;
            letter-spacing: 0.04em;
        }

        .header p {
            font-size: 0.9em;
            color: #6b7280;
            margin-top: 4px;
        }

        .chart-wrap {
            background: rgba(10, 12, 20, 0.92);
            border: 1px solid rgba(55, 60, 90, 0.5);
            border-radius: 12px;
            padding: 28px 28px 20px 28px;
            margin-bottom: 28px;
        }

        .chart-wrap canvas {
            display: block;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }

        .stat-card {
            background: rgba(20, 22, 35, 0.9);
            border: 1px solid rgba(55, 60, 90, 0.45);
            border-radius: 10px;
            padding: 18px 20px;
        }

        .stat-label {
            font-size: 0.78em;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 1.8em;
            font-weight: 700;
            color: #a5b4fc;
        }

        .table-wrap {
            background: rgba(20, 22, 35, 0.9);
            border: 1px solid rgba(55, 60, 90, 0.45);
            border-radius: 10px;
            overflow: hidden;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875em;
        }

        th {
            background: rgba(10, 12, 20, 0.8);
            color: #8892aa;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.78em;
            letter-spacing: 0.06em;
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid rgba(55, 60, 90, 0.5);
        }

        td {
            padding: 10px 16px;
            border-bottom: 1px solid rgba(38, 42, 62, 0.6);
            color: #c8cfe0;
        }

        tr:last-child td { border-bottom: none; }

        tr:hover td { background: rgba(55, 60, 90, 0.15); }

        .speedup-good      { color: #68d391; font-weight: 600; }
        .speedup-excellent { color: #0a0c14; background: #68d391; padding: 2px 7px; border-radius: 4px; font-weight: 700; }

        .footer {
            text-align: center;
            padding: 16px 0 4px;
            color: #4b5563;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
<div class="page">
    <div class="header">
        <h1>ARM NEON — Performance Analysis</h1>
        <p>Сравнение скалярной и NEON векторной обработки массивов · время в миллисекундах · логарифмический масштаб по N</p>
    </div>

    <div class="chart-wrap">
        <div style="position:relative;height:380px;">
            <canvas id="mainChart"></canvas>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Среднее ускорение NEON</div>
            <div class="stat-value" id="avgSpeedupNeon">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Максимальное ускорение</div>
            <div class="stat-value" id="maxSpeedup">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Тестовых точек</div>
            <div class="stat-value" id="totalTests">—</div>
        </div>
    </div>

    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Размер массива</th>
                    <th>Скалярная (мс)</th>
                    <th>NEON (мс)</th>
                    <th>Ускорение NEON</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>

    <div class="footer">Сгенерировано: )";

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    html << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");

    html << R"( · ARM NEON Benchmark</div>
</div>

<script>
    const RAW = [)";

    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) html << ",";
        html << "\n        {";
        html << "size:" << results[i].size << ",";
        html << "scalar:" << std::fixed << std::setprecision(6) << (results[i].scalar_time / 1e3) << ",";
        html << "neon:"   << (results[i].neon_time    / 1e3) << ",";
        html << "unrolled:" << (results[i].unrolled_time / 1e3) << ",";
        html << "speedupNeon:"     << results[i].speedup_neon     << ",";
        html << "speedupUnrolled:" << results[i].speedup_unrolled;
        html << "}";
    }

    html << R"(
    ];

    const glowPlugin = {
        id: 'glow',
        beforeDatasetDraw(chart, args) {
            const ctx = chart.ctx;
            ctx.save();
            ctx.shadowBlur  = 18;
            ctx.shadowColor = args.meta.dataset.options.borderColor;
        },
        afterDatasetDraw(chart) {
            chart.ctx.restore();
        }
    };

    const COLORS = [
        { line: 'rgba(245,101,101,1)',   glow: 'rgba(245,101,101,0.5)'  },
        { line: 'rgba(72,187,120,1)',    glow: 'rgba(72,187,120,0.5)'   },
    ];

    const LABELS   = ['Скалярная', 'NEON'];
    const KEYS     = ['scalar', 'neon'];

    const xTicks = [1000, 10000, 100000, 1000000, 10000000];

    const DATA = RAW.filter(r => r.size >= 1000);

    const datasets = KEYS.map((key, i) => ({
        label: LABELS[i],
        data: DATA.map(r => ({ x: r.size, y: r[key] })),
        borderColor: COLORS[i].line,
        backgroundColor: 'transparent',
        borderWidth: 2.2,
        pointRadius: 0,
        pointHoverRadius: 5,
        tension: 0.35,
    }));

    const ctx = document.getElementById('mainChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: { datasets },
        plugins: [glowPlugin],
        options: {
            responsive: true,
            animation: false,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'start',
                    labels: {
                        color: '#8892aa',
                        font: { size: 13, family: "'Segoe UI', sans-serif" },
                        boxWidth: 28,
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'line',
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10,12,20,0.95)',
                    borderColor: 'rgba(55,60,90,0.7)',
                    borderWidth: 1,
                    titleColor: '#a5b4fc',
                    bodyColor: '#c8cfe0',
                    padding: 12,
                    callbacks: {
                        title: items => 'N = ' + Number(items[0].parsed.x).toLocaleString('ru-RU'),
                        label: item => ' ' + item.dataset.label + ': ' + item.parsed.y.toFixed(4) + ' мс'
                    }
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    grid: { color: 'rgba(38,42,62,0.8)' },
                    border: { color: 'rgba(90,95,125,0.8)' },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 12 },
                        maxRotation: 0,
                        includeBounds: false,
                        callback: function(value) {
                            return xTicks.includes(value) ? value.toLocaleString('ru-RU') : null;
                        }
                    },
                    afterBuildTicks(axis) {
                        axis.ticks = xTicks
                            .filter(v => v >= axis.min && v <= axis.max)
                            .map(v => ({ value: v }));
                    },
                    title: {
                        display: true,
                        text: 'Размер массива (элементы)',
                        color: '#6b7280',
                        font: { size: 12 }
                    }
                },
                y: {
                    type: 'linear',
                    grid: { color: 'rgba(38,42,62,0.8)' },
                    border: { color: 'rgba(90,95,125,0.8)' },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 12 },
                        callback: value => value.toFixed(3) + ' мс'
                    },
                    title: {
                        display: true,
                        text: 'Время (мс)',
                        color: '#6b7280',
                        font: { size: 12 }
                    }
                }
            }
        }
    });

    const speedupsNeon = RAW.map(r => r.speedupNeon);

    const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

    document.getElementById('avgSpeedupNeon').textContent = avg(speedupsNeon).toFixed(2) + 'x';
    document.getElementById('maxSpeedup').textContent     = Math.max(...speedupsNeon).toFixed(2) + 'x';
    document.getElementById('totalTests').textContent         = RAW.length;

    const tbody = document.getElementById('tableBody');
    DATA.forEach(r => {
        const fmt = v => v.toFixed(4) + ' мс';
        const cls = v => v >= 4 ? 'speedup-excellent' : 'speedup-good';
        const row = document.createElement('tr');
        row.innerHTML =
            `<td><strong>${r.size.toLocaleString('ru-RU')}</strong></td>` +
            `<td>${fmt(r.scalar)}</td>` +
            `<td>${fmt(r.neon)}</td>` +
            `<td class="${cls(r.speedupNeon)}">${r.speedupNeon.toFixed(2)}x</td>`;
        tbody.appendChild(row);
    });
</script>
</body>
</html>
)";

    html.close();
    std::cout << "\n✓ HTML график сохранен: " << path << "\n";
}

BenchmarkResult benchmark(size_t array_size) {
    alignas(16) std::vector<int32_t> data(array_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);

    for (size_t i = 0; i < array_size; ++i) {
        data[i] = dist(gen);
    }

    const int iterations = (array_size < 10000) ? 1000 : 100;

    auto start_scalar = std::chrono::high_resolution_clock::now();
    int64_t result_scalar = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        result_scalar = process_array_scalar(data.data(), array_size);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double duration_scalar = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_scalar - start_scalar).count() / (double)iterations / 1000.0;

    auto start_neon = std::chrono::high_resolution_clock::now();
    int64_t result_neon = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        result_neon = process_array_neon(data.data(), array_size);
    }
    auto end_neon = std::chrono::high_resolution_clock::now();
    double duration_neon = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_neon - start_neon).count() / (double)iterations / 1000.0;

    auto start_unrolled = std::chrono::high_resolution_clock::now();
    int64_t result_unrolled = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        result_unrolled = process_array_neon_unrolled(data.data(), array_size);
    }
    auto end_unrolled = std::chrono::high_resolution_clock::now();
    double duration_unrolled = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_unrolled - start_unrolled).count() / (double)iterations / 1000.0;

    if (result_scalar != result_neon || result_neon != result_unrolled) {
        std::cerr << "ОШИБКА! Результаты не совпадают для размера " << array_size << "\n";
        std::cerr << "  Скалярная: " << result_scalar << "\n";
        std::cerr << "  NEON: " << result_neon << "\n";
        std::cerr << "  NEON Unrolled: " << result_unrolled << "\n";
    }

    BenchmarkResult result;
    result.size = array_size;
    result.scalar_time = duration_scalar;
    result.neon_time = duration_neon;
    result.unrolled_time = duration_unrolled;
    result.speedup_neon = duration_scalar / duration_neon;
    result.speedup_unrolled = duration_scalar / duration_unrolled;
    result.result_value = result_scalar;

    return result;
}

int main() {
#if HAVE_NEON
    std::cout << "Платформа: ARM NEON (аппаратное ускорение активно)\n\n";
#else
    std::cout << "Платформа: не-ARM (NEON функции используют скалярный fallback)\n\n";
#endif

    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ARM NEON: Эффективная обработка массива с векторизацией     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Запуск тестов производительности...\n";

    std::vector<size_t> test_sizes = {
        4, 8, 16, 32, 64, 128, 256, 512,
        1000, 2000, 3000, 4000, 5000, 7500,
        10000, 15000, 20000, 30000, 40000, 50000,
        75000, 100000, 150000, 200000, 300000, 400000, 500000,
        750000, 1000000, 1500000, 2000000, 10000000
    };

    std::vector<BenchmarkResult> results;

    int progress = 0;
    for (size_t size : test_sizes) {
        progress++;
        std::cout << "\r[" << progress << "/" << test_sizes.size() << "] "
                  << "Тестирование массива размером " << size << " элементов...     " << std::flush;

        results.push_back(benchmark(size));
    }

    std::cout << "\n\n✓ Все тесты завершены!\n\n";

    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Краткая статистика                                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

    double avg_speedup_neon = 0;
    double avg_speedup_unrolled = 0;
    double max_speedup = 0;

    for (const auto& r : results) {
        avg_speedup_neon += r.speedup_neon;
        avg_speedup_unrolled += r.speedup_unrolled;
        max_speedup = std::max(max_speedup, std::max(r.speedup_neon, r.speedup_unrolled));
    }

    avg_speedup_neon /= results.size();
    avg_speedup_unrolled /= results.size();

    std::cout << "Среднее ускорение NEON:          " << std::fixed << std::setprecision(2)
              << avg_speedup_neon << "x\n";
    std::cout << "Среднее ускорение NEON Unrolled: " << avg_speedup_unrolled << "x\n";
    std::cout << "Максимальное ускорение:          " << max_speedup << "x\n";
    std::cout << "Всего тестовых точек:            " << results.size() << "\n\n";

#if HAVE_NEON
    if (avg_speedup_neon >= 3.0) {
        std::cout << "✓ Требование (>3x ускорение) выполнено!\n\n";
    } else {
        std::cout << "⚠ Ускорение меньше требуемого 3x\n\n";
    }
#endif

    std::cout << "Генерация интерактивного HTML графика...\n";
    generate_html_chart(results);

    std::cout << "\nПопытка открыть график в браузере...\n";

    std::string html_path = "neon_benchmark_chart.html";
    int result_open = -1;

#if defined(_WIN32)
    result_open = system(("start \"\" \"" + html_path + "\"").c_str());
#elif defined(__APPLE__)
    result_open = system(("open " + html_path + " 2>/dev/null").c_str());
#else
    result_open = system(("xdg-open " + html_path + " 2>/dev/null &").c_str());
    if (result_open != 0)
        result_open = system(("firefox " + html_path + " 2>/dev/null &").c_str());
    if (result_open != 0)
        result_open = system(("chromium-browser " + html_path + " 2>/dev/null &").c_str());
    if (result_open != 0)
        result_open = system(("google-chrome " + html_path + " 2>/dev/null &").c_str());
#endif

    if (result_open == 0) {
        std::cout << "✓ График открыт в браузере!\n";
    } else {
        std::cout << "⚠ Не удалось автоматически открыть браузер.\n";
        std::cout << "Откройте файл вручную: " << html_path << "\n";
    }

    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Демонстрация ключевых особенностей ARM                       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

#if HAVE_NEON
    std::cout << "✓ SIMD (NEON): обработка 4-8 элементов параллельно\n";
    std::cout << "✓ Баррельный шифтер: vshrq_n_s32 для вычисления модуля\n";
    std::cout << "✓ Безветвевые вычисления: маски вместо if/else\n";
    std::cout << "✓ Множественная загрузка: vld1q_s32 (128 бит за раз)\n";
    std::cout << "✓ Выравнивание памяти: alignas(16) для оптимизации\n";
    std::cout << "✓ Разворачивание цикла: обработка 8 элементов за итерацию\n\n";
#else
    std::cout << "ℹ NEON недоступен на этой платформе — все функции работают в скалярном режиме.\n\n";
#endif

    return 0;
}
