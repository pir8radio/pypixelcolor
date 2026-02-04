"""
Password authentication command for iPixelColor LED signs.
"""

from ..lib.command_result import CommandResult

CMD_PASSWORD = 0xA1


async def set_password(session, password: str):
    """
    Send password authentication packet (CMD 0xA1).

    Args:
        session: DeviceSession instance
        password: ASCII password string
    """
    if not password:
        return CommandResult(success=True, data=None)

    payload = password.encode("ascii")

    # Use the same low-level helper used by other simple commands
    result = await session.send_simple_command(CMD_PASSWORD, payload)

    return CommandResult(success=result.success, data=None)
