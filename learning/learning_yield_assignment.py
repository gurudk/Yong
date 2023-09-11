
def f():
    i = 0
    while True:
        print(f"[Sub]Begin yield ,第{i}次循环")
        given = yield
        print(f"[Sub]End   yield ,第{i}次循环")
        print(f"[Sub] Receive the send signal,第{i}次循环")
        print(f"[Sub]You send me:{given},第{i}次循环")
        i += 1


print("[Main]Before g generator..")
g = f()
print("[Main]After g generator..")
print("[Main]Before next(g) ..")
next(g)
print("[Main]After next(g)..")
print("[Main]Prepare to send Hi..")
g.send("Hi")
print("[Main]After send Hi..")
print("[Main]Prepare to send Hello..")
g.send("Hello")
print("[Main]After send Hello..")
print("[Main]Prepare to send : How are you!")
g.send("How are you!")
print("[Main]After send : How are you!")

