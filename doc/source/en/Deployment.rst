Deployment
==========

Switch language: :doc:`../zh/Deployment`

Documentation build model
+++++++++++++++++++++++++

The repository uses Sphinx source files under ``doc/source``. The intended
publication workflow is:

1. maintain bilingual ``.rst`` pages locally in the repository;
2. build static HTML with Sphinx;
3. publish the generated site to GitHub Pages.

Why GitHub Pages
++++++++++++++++

GitHub Pages fits this project well because:

- the documentation lives beside the source code;
- no separate documentation repository is required;
- the site URL is predictable under ``github.io``;
- project updates and documentation updates can remain version-aligned.

Expected site URL
+++++++++++++++++

For a normal project repository, the public site will usually look like:

``https://<github-username>.github.io/Enhancer-Promoter-Interaction/``

Build options
+++++++++++++

There are two common publication patterns:

- build locally and publish the generated HTML;
- build automatically in GitHub Actions and deploy from the workflow.

The second option is usually cleaner for long-term maintenance because the
repository stores source documentation rather than generated artifacts.

Suggested deployment path
+++++++++++++++++++++++++

For this repository, the most practical approach is:

- keep Sphinx source in ``doc/``;
- let GitHub Actions build the docs;
- deploy the built output to GitHub Pages;
- add the final site URL to ``README.md`` after deployment is confirmed.

Account and permission requirements
+++++++++++++++++++++++++++++++++++

To publish through GitHub Pages, the repository owner or a collaborator with
sufficient permissions needs:

- a GitHub account;
- access to the target repository;
- permission to enable Pages and, if needed, configure Actions deployment.

What is not required
++++++++++++++++++++

- a separate repository is not required;
- a paid hosting platform is not required for basic documentation hosting;
- a special documentation platform such as Read the Docs is not required.

Practical next step
+++++++++++++++++++

Once the content is finalized, the next implementation step is to add a build
and deployment workflow, then publish the generated HTML and place the final URL
in the project README.

.. image:: ../img/div.png

