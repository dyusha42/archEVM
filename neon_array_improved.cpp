#include <arm_neon.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        
        if (val > 0) {
            sum += val;
        } else if (val < 0) {
            sum += -val;
        }
    }
    
    return sum;
}

int64_t process_array_neon(const int32_t* data, size_t n) {
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
}

int64_t process_array_neon_unrolled(const int32_t* data, size_t n) {
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

std::string get_html_path() {
    #if defined(__aarch64__) || defined(__arm__)
        return "/home/dyusha/archEVM/neon_benchmark_chart.html";
    #else
        return "/home/dyusha/archEVM/neon_benchmark_chart.html";
    #endif
}

void generate_html_chart(const std::vector<BenchmarkResult>& results) {
    std::string path = get_html_path();
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
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: #f7fafc;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .chart-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        
        .chart-title {
            font-size: 1.3em;
            color: #2d3748;
            margin-bottom: 15px;
            font-weight: 600;
            text-align: center;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: scale(1.05);
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
        }
        
        .table-container {
            margin-top: 30px;
            overflow-x: auto;
            background: #f7fafc;
            border-radius: 15px;
            padding: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        th {
            background: #2d3748;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background: #edf2f7;
        }
        
        .speedup-good {
            color: #38a169;
            font-weight: bold;
        }
        
        .speedup-excellent {
            color: #2d3748;
            font-weight: bold;
            background: #68d391;
            padding: 3px 8px;
            border-radius: 5px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ARM NEON Performance Analysis</h1>
            <p>Сравнение скалярной и векторной обработки массивов</p>
        </div>
        
        <div class="content">
            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">Время выполнения (логарифмическая шкала)</div>
                    <canvas id="timeChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Ускорение относительно скалярной версии</div>
                    <canvas id="speedupChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container" style="margin-bottom: 30px;">
                <div class="chart-title">Производительность (элементов/микросекунда)</div>
                <canvas id="throughputChart"></canvas>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Среднее ускорение NEON</div>
                    <div class="stat-value" id="avgSpeedupNeon">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Среднее ускорение NEON Unrolled</div>
                    <div class="stat-value" id="avgSpeedupUnrolled">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Максимальное ускорение</div>
                    <div class="stat-value" id="maxSpeedup">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Всего тестовых точек</div>
                    <div class="stat-value" id="totalTests">-</div>
                </div>
            </div>
            
            <div class="table-container">
                <table id="resultsTable">
                    <thead>
                        <tr>
                            <th>Размер массива</th>
                            <th>Скалярная (мкс)</th>
                            <th>NEON (мкс)</th>
                            <th>NEON Unrolled (мкс)</th>
                            <th>Ускорение NEON</th>
                            <th>Ускорение Unrolled</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            Сгенерировано: )";
    
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    html << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    
    html << R"( | ARM NEON Benchmark Tool
        </div>
    </div>
    
    <script>
        const results = [)";
    
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) html << ",";
        html << "\n            {";
        html << "size:" << results[i].size << ",";
        html << "scalar:" << std::fixed << std::setprecision(3) << results[i].scalar_time << ",";
        html << "neon:" << results[i].neon_time << ",";
        html << "unrolled:" << results[i].unrolled_time << ",";
        html << "speedupNeon:" << results[i].speedup_neon << ",";
        html << "speedupUnrolled:" << results[i].speedup_unrolled;
        html << "}";
    }
    
    html << R"(
        ];
        
        const sizes = results.map(r => r.size);
        const scalarTimes = results.map(r => r.scalar);
        const neonTimes = results.map(r => r.neon);
        const unrolledTimes = results.map(r => r.unrolled);
        const speedupsNeon = results.map(r => r.speedupNeon);
        const speedupsUnrolled = results.map(r => r.speedupUnrolled);
        
        const throughputScalar = results.map(r => r.size / r.scalar);
        const throughputNeon = results.map(r => r.size / r.neon);
        const throughputUnrolled = results.map(r => r.size / r.unrolled);
        
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 12,
                            family: "'Segoe UI', sans-serif"
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 13
                    }
                }
            }
        };
        
        const timeCtx = document.getElementById('timeChart').getContext('2d');
        new Chart(timeCtx, {
            type: 'line',
            data: {
                labels: sizes,
                datasets: [
                    {
                        label: 'Скалярная',
                        data: scalarTimes,
                        borderColor: '#f56565',
                        backgroundColor: 'rgba(245, 101, 101, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'NEON',
                        data: neonTimes,
                        borderColor: '#48bb78',
                        backgroundColor: 'rgba(72, 187, 120, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'NEON Unrolled',
                        data: unrolledTimes,
                        borderColor: '#4299e1',
                        backgroundColor: 'rgba(66, 153, 225, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Размер массива (элементы)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Время (мкс)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    }
                }
            }
        });
        
        const speedupCtx = document.getElementById('speedupChart').getContext('2d');
        new Chart(speedupCtx, {
            type: 'bar',
            data: {
                labels: sizes,
                datasets: [
                    {
                        label: 'NEON',
                        data: speedupsNeon,
                        backgroundColor: 'rgba(72, 187, 120, 0.8)',
                        borderColor: '#48bb78',
                        borderWidth: 2
                    },
                    {
                        label: 'NEON Unrolled',
                        data: speedupsUnrolled,
                        backgroundColor: 'rgba(66, 153, 225, 0.8)',
                        borderColor: '#4299e1',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Размер массива',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Ускорение (x раз)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(1) + 'x';
                            }
                        }
                    }
                }
            }
        });
        
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        new Chart(throughputCtx, {
            type: 'line',
            data: {
                labels: sizes,
                datasets: [
                    {
                        label: 'Скалярная',
                        data: throughputScalar,
                        borderColor: '#f56565',
                        backgroundColor: 'rgba(245, 101, 101, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'NEON',
                        data: throughputNeon,
                        borderColor: '#48bb78',
                        backgroundColor: 'rgba(72, 187, 120, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'NEON Unrolled',
                        data: throughputUnrolled,
                        borderColor: '#4299e1',
                        backgroundColor: 'rgba(66, 153, 225, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Размер массива',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Производительность (элементы/мкс)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    }
                }
            }
        });
        
        const avgSpeedupNeon = (speedupsNeon.reduce((a,b) => a+b, 0) / speedupsNeon.length).toFixed(2);
        const avgSpeedupUnrolled = (speedupsUnrolled.reduce((a,b) => a+b, 0) / speedupsUnrolled.length).toFixed(2);
        const maxSpeedup = Math.max(...speedupsNeon, ...speedupsUnrolled).toFixed(2);
        
        document.getElementById('avgSpeedupNeon').textContent = avgSpeedupNeon + 'x';
        document.getElementById('avgSpeedupUnrolled').textContent = avgSpeedupUnrolled + 'x';
        document.getElementById('maxSpeedup').textContent = maxSpeedup + 'x';
        document.getElementById('totalTests').textContent = results.length;
        
        const tableBody = document.getElementById('tableBody');
        results.forEach(r => {
            const row = document.createElement('tr');
            
            const speedupNeonClass = r.speedupNeon >= 4 ? 'speedup-excellent' : 'speedup-good';
            const speedupUnrolledClass = r.speedupUnrolled >= 4 ? 'speedup-excellent' : 'speedup-good';
            
            row.innerHTML = `
                <td><strong>${r.size.toLocaleString()}</strong></td>
                <td>${r.scalar.toFixed(2)}</td>
                <td>${r.neon.toFixed(2)}</td>
                <td>${r.unrolled.toFixed(2)}</td>
                <td class="${speedupNeonClass}">${r.speedupNeon.toFixed(2)}x</td>
                <td class="${speedupUnrolledClass}">${r.speedupUnrolled.toFixed(2)}x</td>
            `;
            
            tableBody.appendChild(row);
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
    
    if (avg_speedup_neon >= 3.0) {
        std::cout << "✓ Требование (>3x ускорение) выполнено!\n\n";
    } else {
        std::cout << "⚠ Ускорение меньше требуемого 3x\n\n";
    }
    
    std::cout << "Генерация интерактивного HTML графика...\n";

    generate_html_chart(results);
    
    std::cout << "\nПопытка открыть график в браузере...\n";
    
    std::string html_path = get_html_path();
    std::string windows_path = "\\\\wsl.localhost\\Ubuntu\\home\\dyusha\\archEVM\\neon_benchmark_chart.html";
    

    int result_open = system(("explorer.exe \"" + windows_path + "\" 2>/dev/null").c_str());
    
    if (result_open != 0) {
        result_open = system(("xdg-open " + html_path + " 2>/dev/null &").c_str());
    }
    if (result_open != 0) {
        result_open = system(("firefox " + html_path + " 2>/dev/null &").c_str());
    }
    if (result_open != 0) {
        result_open = system(("chromium-browser " + html_path + " 2>/dev/null &").c_str());
    }
    if (result_open != 0) {
        result_open = system(("google-chrome " + html_path + " 2>/dev/null &").c_str());
    }
    
    if (result_open == 0) {
        std::cout << "✓ График открыт в браузере!\n";
    } else {
        std::cout << "⚠ Не удалось автоматически открыть браузер.\n";
        std::cout << "Откройте файл вручную: " << html_path << "\n";
    }
        
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Демонстрация ключевых особенностей ARM                       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "✓ SIMD (NEON): обработка 4-8 элементов параллельно\n";
    std::cout << "✓ Баррельный шифтер: vshrq_n_s32 для вычисления модуля\n";
    std::cout << "✓ Безветвевые вычисления: маски вместо if/else\n";
    std::cout << "✓ Множественная загрузка: vld1q_s32 (128 бит за раз)\n";
    std::cout << "✓ Выравнивание памяти: alignas(16) для оптимизации\n";
    std::cout << "✓ Разворачивание цикла: обработка 8 элементов за итерацию\n\n";
    
    return 0;
}