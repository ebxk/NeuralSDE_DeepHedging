"""Launch the frozen experiment through Windows' independent process provider."""
import ctypes
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

def in_job(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000,False,pid)
    flag = ctypes.c_int()
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.IsProcessInJob(handle,None,ctypes.byref(flag)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(flag.value)
    finally:
        kernel.CloseHandle(handle)

def write(path,data):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    temporary.replace(path)

def main():
    project=Path.cwd()
    program=Path(__file__).resolve().parent
    out=project/"工作区/数据/处理后/扩展组合实验"
    import os

    old=out/"runner.json"
    if old.exists():
        previous=json.loads(old.read_text(encoding="utf-8-sig"))
        # Do not replace an experiment that is still alive.
        result=subprocess.run(["tasklist","/FI",f"PID eq {previous['pid']}","/FO","CSV","/NH"],
                              capture_output=True,text=True)
        if f'"{previous["pid"]}"' in result.stdout:
            raise RuntimeError("The previous experiment process is still alive")
    tag=datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    stdout=out/f"{tag}.stdout.log"
    stderr=out/f"{tag}.stderr.log"
    command=["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",
             str(program/"run_extended.ps1"),"-OptionFile",
             str(project.parent/"开题/test/510050.XSHG_option_full.csv"),
             "-SpotFile","D:/life/study/quant/投资/数据/ETF/510050_华夏上证50ETF.csv",
             "-Workers","8","-Publish"]
    with stdout.open("wb") as output,stderr.open("wb") as error:
        process=subprocess.Popen(command,cwd=project,stdout=output,stderr=error,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    record=dict(pid=process.pid,supervisor_pid=os.getpid(),stdout=str(stdout),stderr=str(stderr),
                start=datetime.now().astimezone().isoformat(),launch="Win32_Process independent provider",
                in_windows_job=in_job(process.pid))
    write(out/"runner.json",record)

    resource=out/f"{tag}.resources.csv"
    with (out/f"{tag}.monitor.stderr.log").open("wb") as error:
        subprocess.Popen(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass",
                          "-File",str(program/"monitor_resources.ps1"),"-RunnerId",str(process.pid),
                          "-Destination",str(resource)],cwd=project,stdout=subprocess.DEVNULL,stderr=error,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    code=process.wait()
    record.update(exit_code=code,finished=datetime.now().astimezone().isoformat())
    write(out/"runner.json",record)

if __name__=="__main__":
    main()
