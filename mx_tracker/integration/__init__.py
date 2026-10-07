"""The integration with lap_vision: jobs driven over HTTP, events delivered back.

`api` is the HTTP surface, `jobs` runs a session's detection, `store` keeps what
must survive a restart, `engine` turns stored crossings into laps, and
`delivery` sends what the store holds to lap_vision until it is acknowledged.
"""
