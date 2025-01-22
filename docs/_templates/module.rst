{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   Classes
   -------
   .. autosummary::
      :template: class.rst
      :toctree: classes
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}



{%- block modules %}
{%- if modules %}
Modules
-------
.. autosummary::
   :toctree:
   :nosignatures:
   :recursive:
   :template: module.rst
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
