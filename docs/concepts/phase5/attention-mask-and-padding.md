# Attention Masks and Padding

## Why Padding Is Needed

A tensor must have a rectangular shape.

Suppose two requests have:

```text
A → 6 real tokens
B → 13 real tokens
```

They cannot directly become one dense tensor with sequence lengths 6 and 13.

So we choose:

\[
L_{max}=13
\]

and pad the shorter request.

## Left Padding

With left padding:

```text
A:
[PAD PAD PAD PAD PAD PAD A0 A1 A2 A3 A4 A5]

B:
[B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12]
```

Both now have length 13.

## Attention Mask

The mask describes which positions are real.

For A:

```text
[0 0 0 0 0 0 1 1 1 1 1 1]
```

For B:

```text
[1 1 1 1 1 1 1 1 1 1 1 1 1]
```

where:

```text
0 → padding / invalid position
1 → real token
```

The exact mask shape and combination with causal masking depend on the model/runtime API.

## The Important Position Idea

Suppose A has 6 real tokens.

The next real token comes after them.

Using zero-based positions:

```text
token 0 → position 0
token 1 → position 1
...
token 5 → position 5
new token → position 6
```

So:

\[
\boxed{
position=6
}
\]

The fact that A was physically padded on the left does not change its logical position.

## Logical Position vs Physical Index

These are different.

Suppose the padded sequence has 13 positions.

The new token can be aligned at the right edge of the dense representation.

Then:

```text
logical position = 6
physical tensor position = last position
```

Therefore:

\[
\boxed{
\text{logical position} \neq \text{padded tensor index}
}
\]

## Why This Matters

Transformers use positional information.

Therefore the runtime must preserve the request's actual sequence position even when the request is padded to a common batch length.

In custom cached decoding code, it is safer to treat logical position or cache position as explicit request metadata rather than assuming padding alone handles every positional detail.

## Mental Model

Think of padding as changing the storage shape, not the meaning:

```text
Logical:
A → A0 A1 A2 A3 A4 A5 NEW

Storage:
A → PAD PAD PAD PAD PAD PAD A0 A1 A2 A3 A4 A5 NEW
```

The storage becomes rectangular, but the logical sequence remains the same.
