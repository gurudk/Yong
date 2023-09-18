"""

第52条 用subprocess管理子进程

"""


import subprocess

result = subprocess.run(
    ['echo', 'Hello from the child'],
    capture_output=True,
    encoding='utf-8'
)

result.check_returncode()
print(result.stdout)


proc = subprocess.Popen(['sleep', '1'])
while proc.poll() is None:
    print('Working')
    print('Exit status', proc.poll())

print(f"[proc.communicate]=======================================================================")
import time

start = time.time()
sleep_procs = []
for _ in range(10):
    proc = subprocess.Popen(['sleep', '1'])
    sleep_procs.append(proc)

for proc in sleep_procs:
    proc.communicate()

end = time.time()
delta = end - start
print(f'Finished in {delta:.3} seconds')

print("[run_encrypt]-----------=================================0=-=----------------===============")
import os


def run_encrpyt(data):
    env = os.environ.copy()
    env['password'] = 'zf7ShyBhZOraQDdE/FiZpm/m/8f9X+M1'
    proc = subprocess.Popen(['openssl','enc','-des3','-pass', 'env:password'],
                            env=env,stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(data)
    proc.stdin.flush()
    return proc


# procs = []
# for _ in range(3):
#     data = os.urandom(10)
#     proc = run_encrpyt(data)
#     procs.append(proc)
#
# for proc in procs:
#     out, _ = proc.communicate()
#     print(out[-10:])


def run_hash(input_stdin):
    return subprocess.Popen(['openssl', 'dgst', '-sha512', '-binary'],
                            stdin=input_stdin, stdout=subprocess.PIPE)


encrypy_procs = []
hash_procs = []

for _ in range(3):
    data = os.urandom(3)
    encrypt_proc = run_encrpyt(data)
    encrypy_procs.append(encrypt_proc)

    hash_proc = run_hash(encrypt_proc.stdout)
    hash_procs.append(hash_proc)

    encrypt_proc.stdout.close()
    encrypt_proc.stdout = None

for proc in encrypy_procs:
    proc.communicate()
    assert proc.returncode == 0

for proc in hash_procs:
    hash_code = proc.communicate()
    print(f'Hashcode:{hash_code}')
    assert proc.returncode == 0

