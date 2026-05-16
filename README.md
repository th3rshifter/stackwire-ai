# StealthWire

## Локальный конфиг

IP сервера берется из `stealthwire.local.env`:

```env
SERVER_IP=[IP]
SERVER_PORT=8000
```

Файл добавлен в `.gitignore`, его можно держать разным на каждом компьютере.

## Запуск

На основном ПК:

```bat
start_server.bat
```

На ноутбуке:

```bat
start_client.bat
```

Можно переопределить IP без редактирования файла:

```bat
start_client.bat [IP]
```
