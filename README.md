# StackWire

## Локальный конфиг

IP сервера берется из `stackwire.local.env`:

```env
SERVER_IP=[IP]
SERVER_PORT=8000
```
<img src="[https://github.com](https://github.com/th3rshifter/stackwire-ai/blob/main/docs/images/1.png)" width="500"
<img src="[https://github.com](https://github.com/th3rshifter/stackwire-ai/blob/main/docs/images/2.png)" width="500"

Файл добавлен в `.gitignore`, его можно держать разным на каждом компьютере.

## Запуск

На основном ПК (серверная часть):

```bat
start_server.bat
```

На клиенте:

```bat
start_client.bat
```

Можно переопределить IP без редактирования файла:

```bat
start_client.bat [IP]
```
