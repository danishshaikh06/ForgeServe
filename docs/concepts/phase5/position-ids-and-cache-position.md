# Position IDs and Cache Position

## Why Position Exists

A transformer needs to know where a token occurs in the sequence.

For a request with:

```text
6 real cached tokens
```

the next token belongs at zero-based position:

\[
6
\]

because the existing positions are:

\[
0,1,2,3,4,5
\]

## Position ID

A position ID answers:

> Where is this token in the logical sequence?

For the next token:

\[
\boxed{
position_i=T_i
}
\]

where \(T_i\) is the number of real tokens already in the sequence.

## Cache Position

A cache position answers a closely related question:

> At which logical sequence position should the new token's K/V be placed in the cache?

For a request with \(T_i\) cached tokens:

\[
\boxed{
cache\_position_i=T_i
}
\]

This is a logical position. It should not be confused with the physical index used by a padded tensor.

## Example

Request A:

```text
6 existing tokens
```

Request B:

```text
13 existing tokens
```

Their next positions are:

```text
A → 6
B → 13
```

Even if both are represented inside a padded tensor, their logical positions remain different.

## Why Left Padding Does Not Change the Logical Position

Suppose:

```text
A has 6 real tokens
Lmax = 13
```

Storage:

```text
[PAD PAD PAD PAD PAD PAD PAD A0 A1 A2 A3 A4 A5]
```

A's next logical position is still:

\[
6
\]

not 12.

The padding exists only because we are using a rectangular batched representation.

## Generation Utilities vs Custom Runtime

Some generation utilities can derive position information from the attention mask.

For a custom inference engine, however, it is useful to keep the request's logical sequence position explicit.

That makes the relationship between:

```text
request state
KV cache
padding
position
```

clear and testable.

## Rule of Thumb

For a decode step:

```text
number of real cached tokens
        ↓
logical next position
        ↓
cache position
        ↓
new token K/V is appended there logically
```

Padding changes physical storage, not the logical position.
