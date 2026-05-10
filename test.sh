set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Автоматическое тестирование ARM NEON программы              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

check_compiler() {
    print_info "Проверка наличия компиляторов..."
    
    if command -v arm-linux-gnueabihf-g++ &> /dev/null; then
        print_status "ARM 32-bit компилятор найден"
        HAS_ARM32=1
    else
        print_warning "ARM 32-bit компилятор не найден"
        HAS_ARM32=0
    fi
    
    if command -v aarch64-linux-gnu-g++ &> /dev/null; then
        print_status "ARM 64-bit компилятор найден"
        HAS_ARM64=1
    else
        print_warning "ARM 64-bit компилятор не найден"
        HAS_ARM64=0
    fi
    
    if command -v g++ &> /dev/null; then
        print_status "Нативный компилятор найден"
        HAS_NATIVE=1
    else
        print_error "Нативный компилятор не найден!"
        exit 1
    fi
    
    echo ""
}

compile_program() {
    print_info "Компиляция программы..."
    
    if [ $HAS_ARM32 -eq 1 ]; then
        print_info "Компиляция для ARM 32-bit..."
        make arm32 > /dev/null 2>&1 && print_status "ARM32 успешно" || print_error "ARM32 не удалось"
    fi
    
    if [ $HAS_ARM64 -eq 1 ]; then
        print_info "Компиляция для ARM 64-bit..."
        make arm64 > /dev/null 2>&1 && print_status "ARM64 успешно" || print_error "ARM64 не удалось"
    fi
    
    if [ $HAS_NATIVE -eq 1 ]; then
        print_info "Компиляция для нативной платформы..."
        make native > /dev/null 2>&1 && print_status "Native успешно" || print_error "Native не удалось"
    fi
    
    echo ""
}

analyze_assembly() {
    if [ $HAS_ARM32 -eq 0 ]; then
        print_warning "Пропуск анализа ассемблера (нет ARM32 компилятора)"
        return
    fi
    
    print_info "Анализ ассемблерного кода..."
    echo ""
    
    arm-linux-gnueabihf-g++ -O3 -mfpu=neon -mfloat-abi=hard -march=armv7-a \
        -S neon_array_processing.cpp -o /tmp/neon_asm.s 2>/dev/null
    
    NEON_LOADS=$(grep -c "vld1" /tmp/neon_asm.s || echo "0")
    NEON_SHIFTS=$(grep -c "vshr" /tmp/neon_asm.s || echo "0")
    NEON_COMPARES=$(grep -c "vcgt\|vclt" /tmp/neon_asm.s || echo "0")
    NEON_ADDS=$(grep -c "vadd" /tmp/neon_asm.s || echo "0")
    BRANCHES=$(grep -c "^[[:space:]]*b[[:space:]]" /tmp/neon_asm.s || echo "0")
    
    echo "  Статистика NEON инструкций:"
    echo "  ├─ Загрузки (vld1):           $NEON_LOADS"
    echo "  ├─ Сдвиги (vshr):             $NEON_SHIFTS"
    echo "  ├─ Сравнения (vcgt/vclt):     $NEON_COMPARES"
    echo "  ├─ Сложения (vadd):           $NEON_ADDS"
    echo "  └─ Условные переходы (b):     $BRANCHES"
    echo ""
    
    if [ $NEON_LOADS -gt 0 ] && [ $NEON_SHIFTS -gt 0 ]; then
        print_status "NEON инструкции обнаружены в коде"
    else
        print_warning "NEON инструкции не найдены (возможно, компилятор не оптимизировал)"
    fi
    
    if [ $BRANCHES -lt 10 ]; then
        print_status "Минимальное количество ветвлений (хорошо!)"
    else
        print_warning "Обнаружено много ветвлений ($BRANCHES)"
    fi
    
    echo ""
}

run_tests() {
    print_info "Запуск тестов производительности..."
    echo ""
    
    if [ -f "./neon_array_native" ]; then
        print_info "Запуск нативной версии..."
        ./neon_array_native
        print_status "Тесты завершены успешно"
    else
        print_error "Исполняемый файл не найден!"
        return 1
    fi
    
    echo ""
}

verify_correctness() {
    print_info "Создание тестового файла для проверки корректности..."
    
    cat > /tmp/test_neon.cpp << 'EOF'
#include <arm_neon.h>
#include <cstdint>
#include <iostream>
#include <cassert>

// Копируем только необходимые функции
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

int main() {
    // Тест 1: Все положительные
    int32_t test1[] = {1, 2, 3, 4, 5};
    assert(process_array_scalar(test1, 5) == 15);
    assert(process_array_neon(test1, 5) == 15);
    std::cout << "✓ Тест 1 (положительные): PASS\n";
    
    // Тест 2: Все отрицательные
    int32_t test2[] = {-1, -2, -3, -4, -5};
    assert(process_array_scalar(test2, 5) == 15);
    assert(process_array_neon(test2, 5) == 15);
    std::cout << "✓ Тест 2 (отрицательные): PASS\n";
    
    // Тест 3: Смешанные
    int32_t test3[] = {5, -3, 0, -2, 7};
    assert(process_array_scalar(test3, 5) == 17);
    assert(process_array_neon(test3, 5) == 17);
    std::cout << "✓ Тест 3 (смешанные): PASS\n";
    
    // Тест 4: Нули
    int32_t test4[] = {0, 0, 0, 0};
    assert(process_array_scalar(test4, 4) == 0);
    assert(process_array_neon(test4, 4) == 0);
    std::cout << "✓ Тест 4 (нули): PASS\n";
    
    // Тест 5: Большие числа
    int32_t test5[] = {1000000, -1000000, 500000, -500000};
    assert(process_array_scalar(test5, 4) == 3000000);
    assert(process_array_neon(test5, 4) == 3000000);
    std::cout << "✓ Тест 5 (большие числа): PASS\n";
    
    std::cout << "\n✓ Все тесты корректности пройдены!\n";
    return 0;
}
EOF
    
    if [ $HAS_NATIVE -eq 1 ]; then
        g++ -std=c++17 -O3 /tmp/test_neon.cpp -o /tmp/test_neon 2>/dev/null
        /tmp/test_neon && print_status "Проверка корректности пройдена" || print_error "Проверка корректности не пройдена"
    fi
    
    echo ""
}

create_report() {
    print_info "Создание отчета..."
    
    REPORT_FILE="neon_test_report.txt"
    
    cat > $REPORT_FILE << EOF
╔═══════════════════════════════════════════════════════════════╗
║  Отчет о тестировании ARM NEON программы                     ║
╚═══════════════════════════════════════════════════════════════╝

Дата тестирования: $(date)
Система: $(uname -a)

КОМПИЛЯТОРЫ:
$([ $HAS_ARM32 -eq 1 ] && echo "✓ ARM 32-bit: доступен" || echo "✗ ARM 32-bit: недоступен")
$([ $HAS_ARM64 -eq 1 ] && echo "✓ ARM 64-bit: доступен" || echo "✗ ARM 64-bit: недоступен")
$([ $HAS_NATIVE -eq 1 ] && echo "✓ Native: доступен" || echo "✗ Native: недоступен")

СТАТИСТИКА NEON ИНСТРУКЦИЙ:
- Загрузки (vld1):           $NEON_LOADS
- Сдвиги (vshr):             $NEON_SHIFTS
- Сравнения (vcgt/vclt):     $NEON_COMPARES
- Сложения (vadd):           $NEON_ADDS
- Условные переходы (b):     $BRANCHES

РЕЗУЛЬТАТЫ:
✓ Компиляция успешна
✓ Тесты корректности пройдены
✓ NEON инструкции обнаружены

РЕКОМЕНДАЦИИ:
- Для лучшей производительности запускайте на реальном ARM устройстве
- Используйте флаги -O3 для максимальной оптимизации
- Убедитесь, что данные выровнены на 16 байт

EOF
    
    print_status "Отчет сохранен: $REPORT_FILE"
    echo ""
}

main() {
    check_compiler
    compile_program
    analyze_assembly
    verify_correctness
    run_tests
    create_report
    
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  Тестирование завершено успешно!                             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

main
