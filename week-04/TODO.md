- Think about: where are the tables stored? Where are the indexes? What happens if two programs try to
write to the same file at the same time? What can PostgreSQL do that SQLite cannot — and when does
that matter?

- Think about: does subprocess.run() block until the command finishes, or does it return immediately?
What is the difference between subprocess.run() and subprocess.Popen()? When would you want asyn-
chronous execution, and when is synchronous enough?

## In your PDF
write a short section covering these three topics. Explain what SQLite is and how it differs
from a server-based database. Explain how subprocess.run() works and whether it is synchronous or
asynchronous. Explain what wget does and what flags you will use. Use your own words — this section is
about demonstrating that you understand the tools, not about producing polished prose.