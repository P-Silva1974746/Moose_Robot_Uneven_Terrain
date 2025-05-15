# moose_gym/test_socket.py
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 5000))
print("Conectado ao controlador!")
