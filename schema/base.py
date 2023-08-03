from pydantic import BaseModel, ConfigDict


def convert_field_to_camel_case(string: str) -> str:
    return "".join(
        word if index == 0 else word.capitalize()
        for index, word in enumerate(string.split("_"))
    )

class BaseSchemaModel(BaseModel):
    model_config = ConfigDict(
        from_attributes = True,
        validate_assignment = True,
        populate_by_name = True,
        # json_encoders = {datetime.datetime: convert_datetime_to_realworld},
        alias_generator = convert_field_to_camel_case
    )