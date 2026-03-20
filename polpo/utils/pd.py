def df_to_nested_dict(df, outer_key, inner_key, value_col):
    return {
        k: g.set_index(inner_key)[value_col].to_dict() for k, g in df.groupby(outer_key)
    }


__all__ = ["df_to_nested_dict"]
