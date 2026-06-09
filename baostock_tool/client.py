"""baostock 客户端封装:统一管理登录登出,提供上下文管理器。"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import baostock as bs

logger = logging.getLogger(__name__)

# 进程内单例,避免重复登录
_lock = threading.Lock()
_login_count = 0
_last_error: Optional[str] = None


def login() -> bool:
    """登录 baostock,成功返回 True。"""
    global _login_count, _last_error
    with _lock:
        # baostock 自身维护了内部会话状态,这里做进程级引用计数防止提前登出
        lg = bs.login()
        if lg.error_code != "0":
            _last_error = lg.error_msg
            logger.error("baostock 登录失败: %s", lg.error_msg)
            return False
        _login_count += 1
        logger.info("baostock 登录成功 (count=%d)", _login_count)
        return True


def logout() -> None:
    """登出 baostock。"""
    global _login_count
    with _lock:
        if _login_count > 0:
            bs.logout()
            _login_count -= 1
            logger.info("baostock 已登出 (count=%d)", _login_count)


def get_last_error() -> Optional[str]:
    return _last_error


@contextmanager
def session(auto_login: bool = True) -> Iterator[None]:
    """会话上下文管理器,确保使用前已登录、退出时不留多余会话。

    用法:
        with session():
            df = fetch_kline(...)
    """
    if _login_count == 0 and auto_login:
        ok = login()
        if not ok:
            raise RuntimeError(f"baostock 登录失败: {_last_error}")
    try:
        yield
    finally:
        # 不主动 logout,因为多模块共用同一会话;进程结束由 atexit 兜底
        pass


def ensure_login() -> None:
    """确保 baostock 已登录,否则抛错。"""
    if _login_count == 0:
        ok = login()
        if not ok:
            raise RuntimeError(f"baostock 登录失败: {_last_error}")


import atexit

atexit.register(lambda: logout() if _login_count > 0 else None)
