# archEVM

sudo apt install build-essential gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user qemu-user-static

chmod +x test.sh

make test

make run

make arm32

qemu-arm -L /usr/arm-linux-gnueabihf ./neon_arm32
