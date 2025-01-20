#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import socket
import threading

class ClientThread(threading.Thread):
    def __init__(self, client_socket, client_address):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.client_address = client_address

    def run(self):
        print(f"Соединение с {self.client_address} открыто")
        while True:
            data = self.client_socket.recv(1024)
            if not data:
                break
            self.client_socket.send(data) 
        self.client_socket.close()
        print(f"Соединение с {self.client_address} закрыто")

def main():
    host = "127.168.27.1" 
    port = 65535  

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"Сервер запущен на {host}:{port}")

    while True:
        client_socket, client_address = server_socket.accept()
        client_thread = ClientThread(client_socket, client_address)
        client_thread.start()

if __name__ == "__main__":
    main()


# In[ ]:


import socket

def main():
    host = "127.168.27.1" 
    port = 65535 
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print(f"Порт {port} открыт")
    except ConnectionRefusedError:
        print(f"Порт {port} закрыт")
        return

    message = input("Введите сообщение: ")

    client_socket.send(message.encode())

    response = client_socket.recv(1024)
    print("Ответ сервера:", response.decode())

    client_socket.close()

if __name__ == "__main__":
    main()

