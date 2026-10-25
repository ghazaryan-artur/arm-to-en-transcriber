# Shared by setup.command and run.command (sourced): close the Terminal window they were opened in
# by a double-click in Finder. Run from an existing Terminal (./run.command), the window is left alone.

# A double-clicked .command runs in a brand-new shell; an existing Terminal's shell is older.
_parent_age() {  # seconds since the parent shell started (ps etime: [[dd-]hh:]mm:ss)
  local t d=0 h=0 m=0 s=0
  t=$(ps -o etime= -p "$PPID" 2>/dev/null | tr -d ' ') || return 1
  [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
  IFS=: read -r -a p <<<"$t"
  case ${#p[@]} in
    3) h=${p[0]}; m=${p[1]}; s=${p[2]} ;;
    2) m=${p[0]}; s=${p[1]} ;;
    *) return 1 ;;
  esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
OPENED_FROM_FINDER=0
if [ "${TERM_PROGRAM:-}" = "Apple_Terminal" ] && age=$(_parent_age) && [ "$age" -le 3 ]; then
  OPENED_FROM_FINDER=1
fi

# Closes this window a moment after the script has exited, from a process detached from the window
# (setsid), so Terminal doesn't ask whether to terminate running processes. No-op outside Finder.
close_window_on_exit() {
  [ "$OPENED_FROM_FINDER" = 1 ] || return 0
  local tty; tty=$(tty) || return 0
  /usr/bin/perl -e 'use POSIX "setsid"; fork and exit; setsid; sleep 1; exec @ARGV' \
    /usr/bin/osascript \
    -e 'on run argv' \
    -e '  tell application "Terminal"' \
    -e '    repeat with w in windows' \
    -e '      repeat with t in tabs of w' \
    -e '        if tty of t is (item 1 of argv) then' \
    -e '          close w' \
    -e '          return' \
    -e '        end if' \
    -e '      end repeat' \
    -e '    end repeat' \
    -e '  end tell' \
    -e 'end run' "$tty" </dev/null >/dev/null 2>&1 &
}
