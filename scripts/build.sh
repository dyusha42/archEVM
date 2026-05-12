#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_DIR}"

C_RESET=""; C_INFO=""; C_OK=""; C_WARN=""; C_ERR=""

info() { echo -e "${C_INFO}[*]${C_RESET} $*"; }
ok()   { echo -e "${C_OK}[+]${C_RESET} $*"; }
warn() { echo -e "${C_WARN}[!]${C_RESET} $*"; }
err()  { echo -e "${C_ERR}[-]${C_RESET} $*"; }

detect_pm() {
    if   command -v apt-get >/dev/null 2>&1; then echo apt
    elif command -v dnf     >/dev/null 2>&1; then echo dnf
    elif command -v pacman  >/dev/null 2>&1; then echo pacman
    elif command -v zypper  >/dev/null 2>&1; then echo zypper
    elif command -v apk     >/dev/null 2>&1; then echo apk
    elif command -v brew    >/dev/null 2>&1; then echo brew
    else echo none
    fi
}

PM="$(detect_pm)"
SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

install_pkgs() {
    local pkgs=("$@")
    [ ${#pkgs[@]} -eq 0 ] && return 0
    info "Установка: ${pkgs[*]}"
    case "$PM" in
        apt)    $SUDO apt-get update -y && $SUDO apt-get install -y "${pkgs[@]}" ;;
        dnf)    $SUDO dnf install -y "${pkgs[@]}" ;;
        pacman) $SUDO pacman -Sy --noconfirm "${pkgs[@]}" ;;
        zypper) $SUDO zypper install -y "${pkgs[@]}" ;;
        apk)    $SUDO apk add "${pkgs[@]}" ;;
        brew)   brew install "${pkgs[@]}" ;;
        *)      warn "Не определён пакетный менеджер. Установите вручную: ${pkgs[*]}"; return 0 ;;
    esac
}

need_install=()

ensure_tool() {
    local tool="$1"; shift
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool найден ($($tool --version 2>/dev/null | head -1))"
    else
        warn "$tool не найден"
        need_install+=("$@")
    fi
}

ensure_tool cmake cmake
ensure_tool make make

if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    warn "C++ компилятор не найден"
    case "$PM" in
        apt)    need_install+=(build-essential) ;;
        dnf)    need_install+=(gcc-c++) ;;
        pacman) need_install+=(base-devel) ;;
        apk)    need_install+=(g++ make) ;;
        *)      need_install+=(g++) ;;
    esac
else
    ok "C++ компилятор найден"
fi

if [ ${#need_install[@]} -gt 0 ]; then
    if [ "$PM" = "none" ]; then
        err "Невозможно автоматически установить: ${need_install[*]}"
        exit 1
    fi
    install_pkgs "${need_install[@]}"
fi

BUILD_DIR="${PROJ_DIR}/build"

GENERATOR=""
GEN_NAME="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="-G Ninja"
    GEN_NAME="Ninja"
fi

if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    cached_gen="$(grep -E '^CMAKE_GENERATOR:' "${BUILD_DIR}/CMakeCache.txt" | head -1 | cut -d= -f2)"
    if [ -n "${cached_gen}" ] && [ "${cached_gen}" != "${GEN_NAME}" ]; then
        warn "Сменился генератор (${cached_gen} -> ${GEN_NAME}), очищаю build/"
        rm -rf "${BUILD_DIR}"
    fi
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

info "Запуск cmake..."
cmake $GENERATOR -DCMAKE_BUILD_TYPE=Release "${PROJ_DIR}"

info "Сборка..."
cmake --build . --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"

ok "Сборка завершена. Бинарь: ${BUILD_DIR}/bin/neon_benchmark"
