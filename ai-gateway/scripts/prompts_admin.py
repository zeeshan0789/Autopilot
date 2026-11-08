"""Manage prompts and their versions (a command line stand in for the dashboard).

Usage (from ai-gateway/):
    python -m scripts.prompts_admin create support --description "Support bot"
    python -m scripts.prompts_admin add-version support --system "You are helpful." --weight 90
    python -m scripts.prompts_admin weights support 1=50 2=50
    python -m scripts.prompts_admin list
"""

import argparse
import asyncio

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db.models import Prompt, PromptVersion
from app.db.session import SessionLocal, engine


async def find_prompt(session, name: str) -> Prompt:
    result = await session.execute(select(Prompt).where(Prompt.name == name))
    prompt = result.scalar_one_or_none()
    if prompt is None:
        raise SystemExit(f"prompt '{name}' does not exist, create it first")
    return prompt


async def create(args) -> None:
    async with SessionLocal() as session:
        session.add(Prompt(name=args.name, description=args.description))
        try:
            await session.commit()
        except IntegrityError:
            raise SystemExit(f"prompt '{args.name}' already exists")
    print(f"prompt '{args.name}' created")


async def add_version(args) -> None:
    async with SessionLocal() as session:
        prompt = await find_prompt(session, args.name)
        latest = (
            await session.execute(
                select(func.max(PromptVersion.version)).where(
                    PromptVersion.prompt_id == prompt.id
                )
            )
        ).scalar()
        number = (latest or 0) + 1
        session.add(
            PromptVersion(
                prompt_id=prompt.id,
                version=number,
                system_prompt=args.system,
                user_template=args.user_template,
                weight=args.weight,
                notes=args.notes,
            )
        )
        await session.commit()
    print(f"'{args.name}' version {number} added with weight {args.weight}")


def parse_weights(items: list[str]) -> dict[int, int]:
    weights = {}
    for item in items:
        try:
            number, weight = item.split("=")
            weights[int(number)] = int(weight)
        except ValueError:
            raise SystemExit(f"bad weight '{item}', use version=weight, for example 1=90")
        if weights[int(number)] < 0:
            raise SystemExit("weights must not be negative")
    return weights


async def set_weights(args) -> None:
    weights = parse_weights(args.weights)
    async with SessionLocal() as session:
        prompt = await find_prompt(session, args.name)
        versions = (
            await session.execute(
                select(PromptVersion).where(PromptVersion.prompt_id == prompt.id)
            )
        ).scalars().all()
        known = {v.version for v in versions}
        for number in weights:
            if number not in known:
                raise SystemExit(f"'{args.name}' has no version {number}")
        for v in versions:
            v.weight = weights.get(v.version, 0)  # versions that are not listed get 0
        await session.commit()
    print(f"weights of '{args.name}' updated (versions not listed now have weight 0)")


async def list_all(args) -> None:
    async with SessionLocal() as session:
        prompts = (await session.execute(select(Prompt).order_by(Prompt.name))).scalars().all()
        if not prompts:
            print("no prompts yet")
        for prompt in prompts:
            versions = (
                await session.execute(
                    select(PromptVersion)
                    .where(PromptVersion.prompt_id == prompt.id)
                    .order_by(PromptVersion.version)
                )
            ).scalars().all()
            total = sum(v.weight for v in versions)
            print(f"{prompt.name}  ({prompt.description or 'no description'})")
            for v in versions:
                share = f"{v.weight / total * 100:.0f}%" if total and v.weight else "not served"
                print(f"    v{v.version}  weight {v.weight:>3} ({share})  {v.system_prompt[:60]!r}")


async def run(args) -> None:
    try:
        await args.func(args)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create")
    p.add_argument("name")
    p.add_argument("--description")
    p.set_defaults(func=create)

    p = sub.add_parser("add-version")
    p.add_argument("name")
    p.add_argument("--system", required=True)
    p.add_argument("--user-template", dest="user_template")
    p.add_argument("--weight", type=int, default=0)
    p.add_argument("--notes")
    p.set_defaults(func=add_version)

    p = sub.add_parser("weights")
    p.add_argument("name")
    p.add_argument("weights", nargs="+", help="for example 1=90 2=10")
    p.set_defaults(func=set_weights)

    p = sub.add_parser("list")
    p.set_defaults(func=list_all)

    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()