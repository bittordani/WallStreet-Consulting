#!/usr/bin/env python3
import sys
from src.rag.rag_query import ask

def main():
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        q = "¿Cómo va Microsoft hoy?"
    print(ask(q))

if __name__ == "__main__":
    main()
